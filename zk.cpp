#include "zk.h"
#include <zookeeper.h>
#include <iostream>
#include <vector>
#include <rapidjson/document.h>

zhandle_t* zhdl;
const int32_t MAX_BUF_LEN = 20000;

void watch(zhandle_t *zh, int type, int state, const char *path,void *watcherCtx) {
    std::cerr << "notified" << std::endl;
}

int32_t init_zk(std::string addr) {
    zhdl = zookeeper_init(addr.c_str(), watch, 1000, NULL, NULL, 0);
    return 0;
}

std::string zk_get(std::string path) {
    char buf[MAX_BUF_LEN];
    int32_t len;
    zoo_get(zhdl, path.c_str(), 0, buf, &len, 0);
    return std::string(buf, (size_t)len);
}

int32_t update_kafka_settings(std::string zk_kafka_path, AllConfig* cfg) {
    String_vector children;
    std::vector<std::string> vc;
    zoo_get_children(zhdl, zk_kafka_path.c_str(), 0, &children);
    for (int32_t i = 0; i < children.count; i++) {
        std::string s = children.data[i];
        vc.push_back(s);
    }
    deallocate_String_vector(&children);

    std::stringstream brokers_s;
    for (std::vector<std::string>::iterator iter = vc.begin(); iter < vc.end(); iter++) {
        std::string json_data = zk_get(*iter);
        if (json_data.size() == 0) {
            std::cerr << "len(data) is 0 for path " << *iter << std::endl;
        }

        rapidjson::Document d;
        d.Parse(json_data);
        std::string host_path = d["host"].GetString();
        int32_t port = d["port"].GetInt();

        brokers_s << host_path << ":" << port << ","
    }

    std::string brokers = brokers_s.str();
    brokers = brokers.substr(0, brokers_s.size() - 1);

    cfg->kafka_in.brokers = brokers;
    cfg->kafka_out.brokers = brokers;
    std::cerr << "brokers = " << brokers << std::endl;

    return 0;
}