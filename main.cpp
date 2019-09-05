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

const char* MODEL_PATH = "../../../model/tensorrt/yolov3_person_16000.model.trt.bin";
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

nvinfer1::ICudaEngine* initEngine(const char* model_path，nvinfer1::IRuntime* runtime) {
    std::ifstream ifile(model_path, std::ios::binary);
    int32_t size;
    ifile >> size;
    model_data = new char[size + 4];
    ifile.read(model_data, size);
    ifile.close();

    ICudaEngine* engine = runtime->deserializeCudaEngine(modelData, modelSize, nullptr);
    return engine;
}

Logger gLogger;

void mark_a_people(cv::Mat canvas, NvDsInferObjectDetectionInfo people) {
    const static cv::Scalar color(0, 255, 255);
    cv::rectangle(canvas, cv::Point(people.left, people.top), cv::Point(people.left + people.width, people.top + people.height), color, 2);
}

int32_t main(int32_t argc, char** argv) {
    fprintf(stderr, "haha\n");
    fflush(stderr);

    // creating image source
    ImageSource src("tcp://10.249.77.88:18964");

    nvinfer1::IRuntime* runtime = createInferRuntime(gLogger);
    nvinfer1::ICudaEngine* engine = initEngine(MODEL_PATH, runtime);
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
        cv::Mat img_resized(input_tensor_height, input_tensor_width, CV_32FC3, buffers.getBuffer(std::string(input_blob_name)));
        //cv::Mat img_resized(input_tensor_height, input_tensor_width, CV_32FC3);
        fprintf(stderr, "[frame %d]resizing from (%d, %d) to (%d, %d)\n", frames, img.rows, img.cols, img_resized.rows, img_resized.cols);
        cv::resize(img, img_resized, img_resized.size(), 0, 0, CV_INTER_CUBIC);

        auto status = ctx->execute(batch_size, buffers.getDeviceBindings().data());
        if (!status) {
            std::cerr << "execution failed!" << std::endl;
            exit(1);
        }

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
