#include "activity_detection.h"

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

ActivityDetector::ActivityDetector(std::string cfg_path, std::string wts_path, std::string name_path, int32_t batch_size, nvinfer1::ILogger logger) {
    this->batch_size = batch_size;
    
    auto builder = nvinfer1::createInferBuilder(logger);
    this->engine = this->init_engine(cfg_path, wts_path, builder);
    this->ctx = this->engine->createExecutionContext();

    this->buffers = new UnifiedBufManager(std::shared_ptr<nvinfer1::ICudaEngine>(engine, InferDeleter()), this->batch_size);

    std::string line;
    std::ifstream ifs(name_path);
    this->names.clear();
    while (std::get_line(ifs, line)) {
        if (line.size() > 1) {
            this->names.push_back(line);
        }
    }
    ifs.close();
}

nvinfer1::ICudaEngine* PeopleDetector::init_engine(std::string cfg_path, std::string weight_path, nvinfer1::IBuilder* builder) {
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

std::vector<LabeledPeople> PeopleDetector::detect(cv::Mat img, std::vector<NvDsInferParseObjectInfo> boxes) {
    std::vector<LabeledPeople> ret;

    for (int32_t off = 0; off < boxes.size(); off += this->batch_size) {
        float* p = (float*)this->buffers.getBuffer(std::string(input_blob_name));
        int32_t cnt = 0
        int32_t stride = input_tensor_width * input_tensor_height * input_tensor_depth;
        for (int32_t i = 0; i < batch_size && off + i < boxes.size();i ++) {
            auto patch = get_patch(img, boxes[i]);
            mat_8u3c_to_darknet_blob(patch, input_tensor_height, input_tensor_width, input_tensor_depth, p + i * stride);
            cnt ++;
        }

        bool status = this->ctx->execute(cnt, this->buffers.getDeviceBindings().data());
        if (!status) {
            std::cerr << "execution failed!" << std::endl;
            return std::vector<LabeledPeople>();    
        }

        float* res = (float*)this->buffers.getBuffer(std::string(output_blob_name));
        for (int32_t k = 0; k < cnt; k++) {
            int32_t max_id = 0;
            for (int32_t i = 0; i < this->names.size(); i++) {
                if (res[i] > res[max_id]) {
                    max_id = i;
                }
            }
            
            LabeledPeople people {
                .loc = boxes[off + k],
                .prob = res[max_id],
                .activity = this->names[max_id],
            };
            ret.push_back(people);

            res += names.size();
        }
    }

    return ret;
}

ActivityDetector::~ActivityDetector() {

}