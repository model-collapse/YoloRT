#include "plugin_factory.h"
#include "yoloPlugins.h"

YOLOPluginFactory::YOLOPluginFactory() {
}

nvinfer1::IPlugin* YOLOPluginFactory::createPlugin(const char *layerName, const void *serialData, size_t serialLength) {
    std::string ln = layerName;
    if (ln == "YoloLayerV3_TRT") {
        return this->deserialize_yolo_v3(serialData, serialLength);
    } else if (ln == "LReLU_TRT") {
        return this->deserialize_leaky_relu(serialData, serialLength);
    }

    std::cerr << "Unsurpported plugin name: " << ln << std::endl;
    return NULL;
}

nvinfer1::IPlugin* YOLOPluginFactory::deserialize_yolo_v3(const void* buf, int32_t size) {
    return new YoloLayerV3(buf, (size_t)size);
}

nvinfer1::IPlugin* YOLOPluginFactory::deserialize_leaky_relu(const void* buf, int32_t size) {
    const float* p = (float*)buf;
    return createLReLUPlugin(p[0]);
}
