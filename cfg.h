#ifndef _CFG_H_
#define _CFG_H_

#include <string>

struct KafkaCFG {
    std::string brokers;
    std::string topic_name;
    std::string group_name;
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
    std::string zk_addr;
    std::string zk_kafka_path;
    std::string fs_addr;
    YoloCFG yolo;
    ActCFG act;
    KafkaCFG kafka_in;
    KafkaCFG kafka_out;
};

int32_t load_config_from_file(std::string path, AllConfig* cfg);

#endif