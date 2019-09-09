#include "people_counting.h"
#include "nvdsparsebbox_Yolo.h"
#include "nvdsinfer_custom_impl.h"
#include "nvdsparsebbox_Yolo.h"

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

PeopleDetector::PeopleDetector(std::string cfg_path, std::string wts_path, int32_t batch_size, nvinfer1::ILogger& logger) {
    this->batch_size = batch_size;
    
    std::vector<std::string> output_blob_names = {
        "yolo_83",
        "yolo_95",
        "yolo_107"
    };

    this->output_blob_names = output_blob_names;

    auto builder = nvinfer1::createInferBuilder(logger);
    this->engine = this->init_engine(cfg_path, wts_path, builder);
    this->ctx = this->engine->createExecutionContext();

    this->buffers = new UnifiedBufManager(std::shared_ptr<nvinfer1::ICudaEngine>(engine, InferDeleter()), batch_size);

    for (int32_t i = 0; i < 3; i++) {
        NvDsInferLayerInfo layer = this->buffers->getLayerInfo(output_blob_names[i]);
        this->layer_info.emplace_back(layer);
    }

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

std::vector<NvDsInferParseObjectInfo> PeopleDetector::detect(cv::Mat img) {
    std::vector<NvDsInferParseObjectInfo> objs;
    std::vector<NvDsInferParseObjectInfo> calib_objs;

    float* p = (float*)this->buffers->getBuffer(std::string(input_blob_name));
    if (NULL == p) {
        std::cerr << "null pointer of input buffer" << std::endl;
    }

    mat_8u3c_to_darknet_blob(img, input_tensor_height, input_tensor_width, input_tensor_depth, p);
    auto status = this->ctx->execute(batch_size, buffers->getDeviceBindings().data());
    if (!status) {
        std::cerr << "execution failed!" << std::endl;
        return objs;    
    }

    NvDsInferNetworkInfo networkInfo {
        .width = input_tensor_width,
        .height = input_tensor_height,
        .channels = 3,
    };

    NvDsInferParseDetectionParams params {
        .numClassesConfigured = NUM_CLASSES_YOLO,
    };
    
    bool res = NvDsInferParseYoloV3(this->layer_info, networkInfo, params, objs, kANCHORS, kMASKS);
    if (!res) {
        std::cerr << "fail to call NvDsInferParseYoloV3" << std::endl;
    }

    float xScale = (float)img.cols / input_tensor_width;
    float yScale = (float)img.rows / input_tensor_height;
    calib_objs.resize(objs.size());
    for (int32_t i = 0; i < objs.size(); i++) {
        NvDsInferParseObjectInfo obj = objs[i];
        NvDsInferParseObjectInfo nf{
            .classId = obj.classId,
            .left = (int)(obj.left * xScale),
            .right = (int)(obj.top * yScale),
            .width = (int)(obj.width * xScale),
            .height = (int)(obj.height * yScale),
            .detectionConfidence = obj.detectionConfidence,
        };

        calib_objs[i] = nf;
    }

    return calib_objs;
}

PeopleDetector::~PeopleDetector() {

}
