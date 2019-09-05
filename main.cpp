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

void mark_a_people(cv::Mat canvas, NvDsInferObjectDetectionInfo people) {
    const static cv::Scalar color(0, 255, 255);
    cv::rectangle(canvas, cv::Point(people.left, people.top), cv::Point(people.left + people.width, people.top + people.height), color, 2);
}

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

    std::vector<NvDsInferLayerInfo> layerInfo;
    for (int32_t i = 0; i < 3; i++) {
        NvDsInferLayerInfo layer = buffers.getLayerInfo(output_blob_names[i]);
        layerInfo.emplace_back(layer);
    }

    std::cerr << "#layers = " << layerInfo.size() << endl;

    static const std::vector<float> kANCHORS = {
        12, 23,  21,  41, 34,  54,  32,  112,  53, 
        78, 75,  120, 58, 196, 104, 288, 184,  322
    };

    static const std::vector<std::vector<int>> kMASKS = {
        {6, 7, 8},
        {3, 4, 5},
        {0, 1, 2}};

    NvDsInferParseDetectionParams params;
    params.numClassesConfigured = NUM_CLASSES_YOLO;

    int32_t frames = 0;
    while (true) {
        frames ++;
        auto img = src.recv();
        cv::Mat img_resized(input_tensor_height, input_tensor_width, CV_8UC3);
        //cv::Mat img_resized(input_tensor_height, input_tensor_width, CV_32FC3);
        fprintf(stderr, "[frame %d]resizing from (%d, %d) to (%d, %d)\n", frames, img.rows, img.cols, img_resized.rows, img_resized.cols);
        cv::resize(img, img_resized, img_resized.size(), 0, 0, CV_INTER_CUBIC);
        cv::Mat img_float;
        img_resized.convertTo(img_float, CV_32FC3);
        img_float /= 255.0;

        std::memcpy(buffers.getBuffer(std::string(input_blob_name)), img_float.data, batch_size * input_tensor_height * input_tensor_width * input_tensor_depth * sizeof(float));

        auto status = ctx->execute(batch_size, buffers.getDeviceBindings().data());
        if (!status) {
            std::cerr << "execution failed!" << std::endl;
            exit(1);
        }

        float* tensor_data = (float*)buffers.getBuffer(std::string(input_blob_name);
        std::cerr << "---------------" << std::endl;
        for (int32_t i = 0; i < 30; i++) {
            std::cerr << "v:" << tensor_data[i] << "\t";
        }
        std::cerr << std::endl;

        NvDsInferNetworkInfo networkInfo;
        networkInfo.width = img.cols;
        networkInfo.height = img.rows;
        networkInfo.channels = 3;
        
        std::vector<NvDsInferParseObjectInfo> objs;
        bool res = NvDsInferParseYoloV3(layerInfo, networkInfo, params, objs, kANCHORS, kMASKS);
        if (!res) {
            std::cerr << "fail to call NvDsInferParseYoloV3" << std::endl;
            exit(1);
        }

	    std::cerr << "[det] " << objs.size() << " were found" << endl;

        for (auto obj : objs) {
            mark_a_people(img, obj);
        }

        char pathBuf[30];
        sprintf(pathBuf, "dump/frame_%d.jpg", frames);
        cv::imwrite(pathBuf, img);
    }
}
