#include "activity_detection.h"
#include "img.h"

static const CLS_THRES = 0.3;

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

ActivityDetector::ActivityDetector(std::string cfg_path, std::string wts_path, std::string name_path, int32_t batch_size, nvinfer1::ILogger& logger) {
    this->batch_size = batch_size;

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
            this->names.push_back(line);
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

cv::Mat get_patch(cv::Mat img, NvDsInferParseObjectInfo box) {
    cv::Rect rect(box.left, box.top, box.width, box.height);
    return img(rect);
}

std::vector<LabeledPeople> ActivityDetector::detect(cv::Mat img, std::vector<NvDsInferParseObjectInfo> boxes) {
    std::vector<LabeledPeople> ret;

    for (int32_t off = 0; off < (int32_t)boxes.size(); off += this->batch_size) {
        float* p = (float*)this->buffers->getBuffer(std::string(input_blob_name));
        int32_t cnt = 0;
        int32_t stride = input_tensor_width * input_tensor_height * input_tensor_depth;
        for (int32_t i = 0; i < batch_size && off + i < (int32_t)boxes.size();i ++) {
            auto patch = get_patch(img, boxes[i]);
            mat_8u3c_to_darknet_blob(patch, input_tensor_height, input_tensor_width, input_tensor_depth, p + i * stride);
            cnt ++;
        }

        bool status = this->ctx->execute(cnt, this->buffers->getDeviceBindings().data());
        if (!status) {
            std::cerr << "execution failed!" << std::endl;
            return std::vector<LabeledPeople>();    
        }

        float* res = (float*)this->buffers->getBuffer(std::string(output_blob_name));
        assert(res != NULL);
        for (int32_t k = 0; k < cnt; k++) {
            std::vector<Activity> activities;
            for (int32_t i = 0; i < (int32_t)this->names.size(); i++) {
                if (res[i] > CLS_THRES) {
                    Activity act {
                        .activity=names[i],
                        .prob=res[i],
                    };

                    activities.append(act);
                }
            }
            
            std::sort(activities.begin(), activities.end(), [](const Activity&a, const Activity&b) {
                return a.prob > b.prob;
            });

            LabeledPeople people {
                .loc = boxes[off + k],
                .activities = activities;
            };
            ret.push_back(people);

            res += names.size();
        }
    }

    return ret;
}

ActivityDetector::~ActivityDetector() {

}
