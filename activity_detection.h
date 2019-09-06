#ifndef _ACT_DET_H_
#define _ACT_DET_H_

struct LabeledPeople struct {
    NvDsInferParseObjectInfo loc;
    std::string activity;
    float prob;
};

class ActivityDetector {
public:
    static const int32_t input_tensor_height = 256;
    static const int32_t input_tensor_width = 256;
    static const int32_t input_tensor_depth = 3;

    static const char* input_blob_name = "data";
    static const char* output_blob_name = "yolo_";

    static const int32_t MAX_BATCH_SIZE = 16;

    ActivityDetector(std::string cfg_path, std::string wts_path, int32_t batch_size, nvinfer1::ILogger logger);
    ~ActivityDetector();

    std::vector<LabeledPeople> detect(cv::Mat img, std::vector<NvDsInferParseObjectInfo> boxes);
private:
    nvinfer1::ICudaEngine* init_engine(std::string cfg_path, std::string weight_path, nvinfer1::IBuilder* builder);
    
    nvinfer1::ICudaEngine* engine;
    nvinfer1::IExecutionContext* ctx;

    UnifiedBufManager* buffers;

    int32_t batch_size;

    std::vector<std::string> names;
}

#endif