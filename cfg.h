#ifndef _CFG_H_
#define _CFG_H_

#include <string>

struct MQTTCFG {
    std::string addr;
    std::string client_name;
    bool clean_session;
    std::string in_topic_name;
    std::string out_topic_name;
};

struct YoloCFG {
    std::string model_file;
    float cls_thres;
    float nms_thres;
};

struct ActCFG {
    std::string model_file;
    std::string name_file;
    int32_t batch_size;
    float   ext_scale;
};

struct AllConfig {
    YoloCFG yolo;
    ActCFG act;
    MQTTCFG mqtt;
};

int32_t load_config_from_file(std::string path, AllConfig* cfg);

#endif