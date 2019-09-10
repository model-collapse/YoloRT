#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <iostream>

int32_t main(int32_t argc, char** argv) {
    nvinfer1::IPluginV2* leakyRELU = createLReLUPlugin(0.1);
    std::cerr << "serialized len = " << leakyRELU->getSerializationSize() << endl;
}