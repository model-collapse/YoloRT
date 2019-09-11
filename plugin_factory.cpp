#include "plugin_factory.h"
#include "yoloPlugins.h"

YOLOPluginFactory::YOLOPluginFactory() {
}

IPlugin* YOLOPluginFactory::createPlugin(const char *layerName, const void *serialData, size_t serialLength) {
    std::string ln = layerName;
    if (ln == "YoloLayerV3_TRT") {
        return this->deserialize_yolo_v3(serialData, serialLength);
    } else if (ln == "LReLU_TRT") {
        return this->deserialize_leaky_relu(serialData, serialLength);
    }

    std::cerr << "Unsurpported plugin name: " << ln << std::endl;
    return NULL;
}

IPlugin* YOLOPluginFactory::deserialize_yolo_v3(void* buf, int32_t size) {
    return new YoloLayerV3(buf, (size_t)size);
}

IPlugin* YOLOPluginFactory::deserialize_leaky_relu(void* buf, int32_t size) {
    float* p = buf;
    return nvinfer1::createLReLUPlugin(p[0]);
}