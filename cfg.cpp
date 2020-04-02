#include "cfg.h"
#include <boost/program_options.hpp>
#include <iostream>
#include <fstream>

int32_t load_config_from_file(std::string path, AllConfig* cfg) {
    namespace po = boost::program_options;
    po::options_description opt("WW");
    opt.add_options()("MQTT.address", po::value(&cfg->mqtt.addr), "MQTT broker address");
    opt.add_options()("MQTT.client_name", po::value(&cfg->mqtt.client_name), "MQTT broker client name");
    opt.add_options()("MQTT.clean_session", po::value(&cfg->mqtt.clean_session), "MQTT use persistent queue");
    opt.add_options()("MQTT.in_topic", po::value(&cfg->mqtt.in_topic_name), "MQTT input topic name");
    opt.add_options()("MQTT.out_topic", po::value(&cfg->mqtt.out_topic_name), "MQTT output topic name");
    opt.add_options()("YoloCFG.model_file", po::value(&cfg->yolo.model_file), "path to yolo model file");
    opt.add_options()("YoloCFG.class_thres", po::value(&cfg->yolo.cls_thres), "threshold of class prob");
    opt.add_options()("YoloCFG.nms_thres", po::value(&cfg->yolo.nms_thres), "threshold of non-maximum suppression");
    opt.add_options()("ActCFG.model_file", po::value(&cfg->act.model_file), "path to action detection model file");
    opt.add_options()("ActCFG.name_file", po::value(&cfg->act.name_file), "path to action detection name file");
    opt.add_options()("ActCFG.batch_size", po::value(&cfg->act.batch_size), "action detection batch size");
    opt.add_options()("ActCFG.ext_scale", po::value(&cfg->act.ext_scale), "action detection bouding box extension scale");

    std::ifstream config_stream(path.c_str());
    po::variables_map vm;
    po::store(po::parse_config_file(config_stream, opt), vm);
    po::notify(vm);

    return 0;
}
