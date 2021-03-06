#ifndef _ACT_DET_H_
#define _ACT_DET_H_

#include <string>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "yolo.h"
#include "jbuf.h"
#include "def.h"
#include "nvdsinfer_custom_impl.h"

struct Activity {
    std::string activity;
    float prob;
};

struct LabeledPeople {
    NvDsInferParseObjectInfo loc;
    std::vector<Activity> activities;
};

class ActivityDetector {
public:
    static const int32_t input_tensor_height = 256;
    static const int32_t input_tensor_width = 256;
    static const int32_t input_tensor_depth = 3;

    static constexpr const char* input_blob_name = "data";
    static constexpr const char* output_blob_name = "softmax_78";

    static const int32_t MAX_BATCH_SIZE = 4;

    ActivityDetector(std::string cfg_path, std::string wts_path, std::string names_path, int32_t batch_size, float ext_scale, nvinfer1::ILogger& logger);
    ActivityDetector(std::string model_path, std::string names_path, int32_t batch_size, float ext_scale, nvinfer1::ILogger& logger);
    ~ActivityDetector();

    std::vector<LabeledPeople> detect(cv::Mat img, std::vector<NvDsInferParseObjectInfo> boxes);
    int32_t detect_capi(cv::Mat img, NvDsInferParseObjectInfo* boxes, int32_t num, const char* res[][NUM_ACTIVITIES]);
private:
    nvinfer1::ICudaEngine* init_engine(std::string cfg_path, std::string weight_path, nvinfer1::IBuilder* builder);
    cv::Mat get_patch(cv::Mat img, NvDsInferParseObjectInfo box);

    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* ctx;

    UnifiedBufManager* buffers;

    int32_t batch_size;
    float ext_scale;

    std::vector<std::string> names;
    std::vector<float> thresholds;
};

#endif
