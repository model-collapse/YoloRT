#include <iostream>
#include "recv.h"
#include "yolo.h"
#include "jbuf.h"

const char* CFG_PATH = "";
const char* WTS_PATH = "";
const int32_t batch_size = 1;

const int32_t input_tensor_height = 640;
const int32_t input_tensor_width = 640;
const int32_t input_tensor_depth = 3;

const char* input_blob_name = "data";

nvinfer1::ICudaEngine* initEngine(const char* cfg_path, const char* weight_path, nvinfer1::IBuilder* builder) {
    nvinfer1::IBuilder* builder = createInferBuilder(gLogger);
    NetworkInfo info;
    info.networkType = "yolov3";
    info.configFilePath = cfg_path;
    info.wtsFilePath = weight_path;
    info.deviceType = "kGPU";
    info.inputBlobName = input_blob_name;
    
    Yolo yolo(info, builder);
    return yolo.createEngine();
}

int32_t main(int32_t argc, char** argv) {
    // creating image source
    ImageSource src("tcp://10.249.77.88");

    nvinfer1::IBuilder* builder = createInferBuilder(gLogger);
    nvinfer1::ICudaEngine* engine = initEngine(CFG_PATH, WTS_PATH, builder);
    auto ctx = createExecutionContext();

    int32_t tensor_size = buffers.size(input_blob_name);
    if (tensor_size != batch_size * input_tensor_height * input_tensor_width * input_tensor_depth * sizeof(float)) {
        std::cerr << "inconsistent input tensor size" << std::endl;
        exit(1);
    }

    UnifiedBufManger buffers(engine, batch_size);

    while (true) {
        auto img = src.recv();
        cv::Mat img_resized(input_tensor_height, input_tensor_width, cv::CV_32FC3, buffers.getBuffer(std::string(inputBlobName)));
        cv::resize(img, img_resized, img_resized.size(), 0, 0, INTER_CUBIC);

        auto status = ctx->execute(batch_size, buffers.getDeviceBindings().data());
        if (!status) {
            std::cerr << "execution failed!" << std::endl;
            exit(1);
        }
    }
}