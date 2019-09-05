#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "recv.h"
#include "yolo.h"
#include "jbuf.h"
#include "nvdsinfer_custom_impl.h"
#include "nvdsparsebbox_Yolo.h"

const char* CFG_PATH = "../../../model/darknet/yolov3_person.cfg";
const char* WTS_PATH = "../../../model/darknet/yolov3_person_16000.weights";
const int32_t batch_size = 1;

const int32_t input_tensor_height = 640;
const int32_t input_tensor_width = 640;
const int32_t input_tensor_depth = 3;

const char* input_blob_name = "data";
const char output_blob_names[][20] = {
    "yolo_83",
    "yolo_95",
    "yolo_107"
};

Logger gLogger;

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

int32_t main(int32_t argc, char** argv) {
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    nvinfer1::ICudaEngine* engine = initEngine(CFG_PATH, WTS_PATH, builder);

    IHostMemory *serializedModel = engine->serialize();
    std::ofstream ofile("yolov3_person_16000.model.trt.bin", std::ios::binary);
    
    int64_t size = serializedModel->size();
    std::cerr << "size = " << size << endl;
    ofile.write((char*)&size, sizeof(size));
    ofile.write((char*)serializedModel->data(), serializedModel->size());
    ofile.close();
    engine->destroy();
}
