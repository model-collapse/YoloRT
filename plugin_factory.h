#include "NvInfer.h"
#include "NvInferPlugin.h"

class YOLOPluginFactory : public nvinfer1::IPluginFactory {
public:
    YOLOPluginFactory();
    nvinfer1::IPlugin* createPlugin (const char *layerName, const void *serialData, size_t serialLength);
private:
    nvinfer1::IPlugin* deserialize_yolo_v3(const void* buf, int32_t size);
    nvinfer1::IPlugin* deserialize_leaky_relu(const void* buf, int32_t size);
};
