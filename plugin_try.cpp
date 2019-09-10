#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <iostream>

int32_t main(int32_t argc, char** argv) {
    nvinfer1::IPluginV2* leakyRELU = createLReLUPlugin(0.1);
    std::cerr << "serialized len = " << leakyRELU->getSerializationSize() << std::endl;
    char* buf = new char[leakyRELU->getSerializationSize()];
    leakyRELU->serialize(buf);
    float* p = (float*)buf;
    for (int32_t i = 0; i < 2; i++) {
        std::cerr << "float@" << i << " = " << p[i] << std::endl;
    };

    int32_t* pp = (int32_t*)buf;
    for (int32_t i = 0; i < 2; i++) {
        std::cerr << "int@" << i << " = " << pp[i] << std::endl;
    };
}
