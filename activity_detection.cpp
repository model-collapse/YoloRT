#include "activity_detection.h"
#include "img.h"
#include <boost/tokenizer.hpp>
#include <boost/foreach.hpp>
#include <chrono>

static const float CLS_THRES = 0.3;

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

std::vector<std::string> string_split(std::string line, std::string sep) {
    boost::tokenizer<boost::char_separator<char> > tokens(line, boost::char_separator<char>(sep.c_str()));
    std::vector<std::string> ret;
    for (auto iter = tokens.begin(); iter != tokens.end(); iter++) {
        ret.push_back(*iter);
    }

    return ret;
}

ActivityDetector::ActivityDetector(std::string cfg_path, std::string wts_path, std::string name_path, int32_t batch_size, float ext_scale, nvinfer1::ILogger& logger) {
    this->batch_size = batch_size;
    this->ext_scale = ext_scale;

    auto builder = nvinfer1::createInferBuilder(logger);
    builder->setMaxBatchSize(MAX_BATCH_SIZE);
    this->engine = this->init_engine(cfg_path, wts_path, builder);
    this->ctx = this->engine->createExecutionContext();
    std::cerr << "max_batch_size = " << this->engine->getMaxBatchSize() << std::endl;
    assert(batch_size <= this->engine->getMaxBatchSize());

    this->buffers = new UnifiedBufManager(std::shared_ptr<nvinfer1::ICudaEngine>(engine, InferDeleter()), this->batch_size);

    std::string line;
    std::ifstream ifs(name_path);
    this->names.clear();
    while (std::getline(ifs, line)) {
        if (line.size() > 1) {
            auto eles = string_split(line, "\t");
            this->names.push_back(eles[0]);
            this->thresholds.push_back(std::stof(eles[1]));
        }
    }
    ifs.close();

    std::cerr << "#classes = " << this->names.size() << std::endl;
}

ActivityDetector::ActivityDetector(std::string model_path, std::string name_path, int32_t batch_size, nvinfer1::ILogger& logger) {
    this->batch_size = batch_size;

    IRuntime* runtime = createInferRuntime(gLogger);
    int64_t length;
    std::ifstream model_file(model_path, std::ios::binary);
    if (!model_file) {
        std::cerr << "cannot open file: " << model_path << std::endl;
        return;
    }

    model_file.read((char*)&length, sizeof(length));
    std::cerr << "data length = " << length << std::endl;
    char *buf = new char[length];
    model_file.read(buf, length);
    model_file.close();

    this->engine = runtime->deserializeCudaEngine(buf, length, NULL);
    delete buf;

    this->ctx = this->engine->createExecutionContext();
    std::cerr << "max_batch_size = " << this->engine->getMaxBatchSize() << std::endl;
    assert(batch_size <= this->engine->getMaxBatchSize());

    this->buffers = new UnifiedBufManager(std::shared_ptr<nvinfer1::ICudaEngine>(engine, InferDeleter()), this->batch_size);

    std::string line;
    std::ifstream ifs(name_path);
    this->names.clear();
    while (std::getline(ifs, line)) {
        if (line.size() > 1) {
            auto eles = string_split(line, "\t");
            this->names.push_back(eles[0]);
            this->thresholds.push_back(std::stof(eles[1]));
        }
    }
    ifs.close();

    std::cerr << "#classes = " << this->names.size() << std::endl;
}

nvinfer1::ICudaEngine* ActivityDetector::init_engine(std::string cfg_path, std::string weight_path, nvinfer1::IBuilder* builder) {
    NetworkInfo info;
    info.networkType = "yolov3";
    info.configFilePath = cfg_path;
    info.wtsFilePath = weight_path;
    info.deviceType = "kGPU";
    info.inputBlobName = input_blob_name;
    
    Yolo yolo(info, builder);
    return yolo.createEngine();
}

cv::Mat ActivityDetector::get_patch(cv::Mat img, NvDsInferParseObjectInfo box) {
    float margin_scale = this->ext_scale / 2;
    int32_t margin_x = (int)(box.width * margin_scale);
    int32_t margin_y = (int)(box.height * margin_scale);

    int32_t x = std::max((int32_t)box.left - margin_x, 0);
    int32_t y = std::max((int32_t)box.top - margin_y, 0);
    int32_t w = std::min((int32_t)box.width + 2 * margin_x, (int32_t)img.cols - x);
    int32_t h = std::min((int32_t)box.height + 2 * margin_y, (int32_t)img.rows - y);
    cv::Rect rect(x, y, w, h);
    return img(rect);
}

std::vector<LabeledPeople> ActivityDetector::detect(cv::Mat img, std::vector<NvDsInferParseObjectInfo> boxes) {
    std::vector<LabeledPeople> ret;

    for (int32_t off = 0; off < (int32_t)boxes.size(); off += this->batch_size) {
        auto beg_buf = std::chrono::system_clock::now();
        float* p = (float*)this->buffers->getBuffer(std::string(input_blob_name));
        int32_t cnt = 0;
        int32_t stride = input_tensor_width * input_tensor_height * input_tensor_depth;
        for (int32_t i = 0; i < batch_size && off + i < (int32_t)boxes.size();i ++) {
            auto patch = get_patch(img, boxes[i]);
            mat_8u3c_to_darknet_blob(patch, input_tensor_height, input_tensor_width, input_tensor_depth, p + i * stride);
            cnt ++;
        }
        auto end_buf = std::chrono::system_clock::now();

        auto beg_exe = std::chrono::system_clock::now();
        bool status = this->ctx->execute(cnt, this->buffers->getDeviceBindings().data());
        if (!status) {
            std::cerr << "execution failed!" << std::endl;
            return std::vector<LabeledPeople>();    
        }
        auto end_exe = std::chrono::system_clock::now();

        auto beg_post = std::chrono::system_clock::now();
        float* res = (float*)this->buffers->getBuffer(std::string(output_blob_name));
        assert(res != NULL);
        for (int32_t k = 0; k < cnt; k++) {
            std::vector<Activity> activities;
            for (int32_t i = 0; i < (int32_t)this->names.size(); i++) {
                if (res[i] * boxes[off + k].detectionConfidence> this->thresholds[i]) {
                    Activity act {
                        .activity=names[i],
                        .prob=res[i],
                    };

                    activities.push_back(act);
                }
            }
            
            std::sort(activities.begin(), activities.end(), [](const Activity&a, const Activity&b) {
                return a.prob > b.prob;
            });

            LabeledPeople people {
                .loc = boxes[off + k],
                .activities = activities,
            };
            ret.push_back(people);
            res += names.size();
        }

        auto end_post = std::chrono::system_clock::now();

        auto msecs = [](std::chrono::system_clock::time_point beg, std::chrono::system_clock::time_point end) -> int {
            return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
        };

        std::cerr << "[AD time cost@" << off << "] | buffer:" << msecs(beg_buf, end_buf) << "ms, exe:" << msecs(beg_exe, end_exe) << "ms, post:" << msecs(beg_post, end_post) << "ms" << std::endl;
    }

    return ret;
}

ActivityDetector::~ActivityDetector() {

}
