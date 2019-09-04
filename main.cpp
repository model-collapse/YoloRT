#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "recv.h"
#include "yolo.h"
#include "jbuf.h"

const char* CFG_PATH = "../../../model/darknet/yolov3_person.cfg";
const char* WTS_PATH = "../../../model/darknet/yolov3_person_16000.weights";
const int32_t batch_size = 1;

const int32_t input_tensor_height = 640;
const int32_t input_tensor_width = 640;
const int32_t input_tensor_depth = 3;

const char* input_blob_name = "data";

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

nvinfer1::ICudaEngine* initEngine(const char* cfg_path, const char* weight_path, nvinfer1::IBuilder* builder) {
    NetworkInfo info;
    info.networkType = "yolov3";
    info.configFilePath = cfg_path;
    info.wtsFilePath = weight_path;
    info.deviceType = "kGPU";
    info.inputBlobName = input_blob_name;
    
    Yolo yolo(info, builder);
    return yolo.createEngine();
}

Logger gLogger;

int32_t main(int32_t argc, char** argv) {
    fprintf(stderr, "haha\n");
    fflush(stderr);

    // creating image source
    ImageSource src("tcp://10.249.77.88:18964");

    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    nvinfer1::ICudaEngine* engine = initEngine(CFG_PATH, WTS_PATH, builder);
    auto ctx = engine->createExecutionContext();

    int32_t bsize = batch_size;
    UnifiedBufManager buffers(std::shared_ptr<nvinfer1::ICudaEngine>(engine, InferDeleter()), bsize);
    int32_t tensor_size = buffers.size(input_blob_name);
    if (tensor_size != batch_size * input_tensor_height * input_tensor_width * input_tensor_depth * sizeof(float)) {
        std::cerr << "inconsistent input tensor size" << std::endl;
        exit(1);
    }


    while (true) {
        auto img = src.recv();
        cv::Mat img_resized(input_tensor_height, input_tensor_width, CV_32FC3, buffers.getBuffer(std::string(input_blob_name)));
        cv::resize(img, img_resized, img_resized.size(), 0, 0, CV_INTER_CUBIC);

        auto status = ctx->execute(batch_size, buffers.getDeviceBindings().data());
        if (!status) {
            std::cerr << "execution failed!" << std::endl;
            exit(1);
        }
    }
}
