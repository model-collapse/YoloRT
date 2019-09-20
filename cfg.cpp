#include "cfg.h"
#include <boost/program_options.hpp>
#include <iostream>
#include <fstream>

int32_t load_config_from_file(std::string path, AllConfig* cfg) {
    namespace po = boost::program_options;
    po::options_description opt("WW");
    opt.add_options()("zookeeper.addr", po::value(&cfg->zk_addr), "Address of zookeeper");
    opt.add_options()("file_server.addr", po::value(&cfg->fs_addr), "Address of file_server");
    opt.add_options()("YoloCFG.model_file", po::value(&cfg->yolo.model_file), "path to yolo model file");
    opt.add_options()("YoloCFG.class_thres", po::value(&cfg->yolo.cls_thres), "threshold of class prob");
    opt.add_options()("YoloCFG.nms_thres", po::value(&cfg->yolo.nms_thres), "threshold of non-maximum suppression");
    opt.add_options()("ActCFG.model_file", po::value(&cfg->act.model_file), "path to action detection model file");
    opt.add_options()("ActCFG.name_file", po::value(&cfg->act.name_file), "path to action detection name file");
    opt.add_options()("ActCFG.batch_size", po::value(&cfg->act.batch_size), "action detection batch size");
    opt.add_options()("ActCFG.ext_scale", po::value(&cfg->act.ext_scale), "action detection bouding box extension scale");
    opt.add_options()("kafaka_in.brokers", po::value(&cfg->kafka_in.brokers), "broker list for kafka");
    opt.add_options()("kafaka_in.topic_name", po::value(&cfg->kafka_in.topic_name), "incomming topic name for kafka");
    opt.add_options()("kafaka_in.group_name", po::value(&cfg->kafka_in.group_name), "incomming group name for kafka");
    opt.add_options()("kafaka_out.brokers", po::value(&cfg->kafka_out.brokers), "broker list for kafka");
    opt.add_options()("kafaka_out.topic_name", po::value(&cfg->kafka_out.topic_name), "outputting topic name for kafka");

    std::ifstream config_stream(path.c_str());
    po::variables_map vm;
    po::store(po::parse_config_file(config_stream, opt), vm);
    po::notify(vm);

    return 0;
}
