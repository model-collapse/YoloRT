#ifndef _ZK_H_
#define _ZK_H_

#include "cfg.h"

int32_t init_zk(std::string addr);
std::string zk_get(std::string path);
int32_t update_kafka_settings(std::string zk_kafka_path, AllConfig* cfg);

#endif