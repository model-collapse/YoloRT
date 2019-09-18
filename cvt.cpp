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

const int32_t MAX_BATCH_SIZE = 4;

const char* pc_input_blob_name = "data";
const char pc_output_blob_names[][20] = {
    "yolo_83",
    "yolo_95",
    "yolo_107"
};

const char* AD_CFG_PATH = "../../../model/darknet/wwdarknet53v2.cfg";
const char* AD_WTS_PATH = "../../../model/darknet/wwdarknet53v2_50000.weights";

const char* ad_input_blob_name = "data";
const char* ad_output_blob_name = "softmax_78";

Logger gLogger;

nvinfer1::ICudaEngine* initEngine(const char* cfg_path, const char* weight_path, const char* input_blob_name, nvinfer1::IBuilder* builder) {
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
    std::cerr << "#DLA core = " << builder->getNbDLACore() << std::endl;
    return 0;

    nvinfer1::ICudaEngine* engine = initEngine(CFG_PATH, WTS_PATH, pc_input_blob_name, builder);

    IHostMemory *serializedModel = engine->serialize();
    std::ofstream ofile("../../../model/tensorRT/yolov3_person.trt.dat", std::ios::binary);
    
    int64_t size = serializedModel->size();
    std::cerr << "size = " << size << endl;
    ofile.write((char*)&size, sizeof(size));
    ofile.write((char*)serializedModel->data(), serializedModel->size());
    ofile.close();
    engine->destroy();
    serializedModel->destroy();

    builder->setMaxBatchSize(MAX_BATCH_SIZE);
    engine = initEngine(AD_CFG_PATH, AD_WTS_PATH, ad_input_blob_name,  builder);
    serializedModel = engine->serialize();

    std::ofstream ofile2("../../../model/tensorRT/wwdarknet53v2.trt.dat", std::ios::binary);
    size = serializedModel->size();
    std::cerr << "size = " << size << endl;
    ofile2.write((char*)&size, sizeof(size));
    ofile2.write((char*)serializedModel->data(), serializedModel->size());
    ofile2.close();
    engine->destroy();
    serializedModel->destroy();
}
