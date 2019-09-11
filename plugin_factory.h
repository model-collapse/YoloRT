#include "NvInfer.h"
#include "NvInferPlugin.h"

class YOLOPluginFactory : IPluginFactory {
public:
    YOLOPluginFactory();
    IPlugin* createPlugin (const char *layerName, const void *serialData, size_t serialLength);
private:
    IPlugin* deserialize_yolo_v3(void* buf, int32_t size);
    IPlugin* deserialize_leaky_relu(void* buf, int32_t size);
};