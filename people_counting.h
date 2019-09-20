#ifndef _PC_H_
#define _PC_H_

#include <string>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "yolo.h"
#include "jbuf.h"
#include "img.h"
#include "nvdsinfer_custom_impl.h"

const std::vector<float> kANCHORS = {
    12, 23,  21,  41, 34,  54,  32,  112,  53, 
    78, 75,  120, 58, 196, 104, 288, 184,  322
};

const std::vector<std::vector<int>> kMASKS = {
    {6, 7, 8},
    {3, 4, 5},
    {0, 1, 2}};

class PeopleDetector {
public:
    static const int32_t input_tensor_height = 640;
    static const int32_t input_tensor_width = 640;
    static const int32_t input_tensor_depth = 3;

    static constexpr const char* input_blob_name = "data";
    std::vector<std::string> output_blob_names;

    PeopleDetector(std::string cfg_path, std::string wts_path, int32_t batch_size, float cls_thres, float nms_thres, nvinfer1::ILogger& logger);
    PeopleDetector(std::string model_path, int32_t batch_size, nvinfer1::ILogger& logger);
    ~PeopleDetector();

    std::vector<NvDsInferParseObjectInfo> detect(cv::Mat img);
private:
    nvinfer1::ICudaEngine* init_engine(std::string cfg_path, std::string weight_path, nvinfer1::IBuilder* builder);
    
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* ctx;

    UnifiedBufManager* buffers;

    std::vector<NvDsInferLayerInfo> layer_info;

    float nms_thres;
    float cls_thes;
    int32_t batch_size;
};

#endif
