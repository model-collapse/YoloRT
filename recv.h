#ifndef _RECV_H_
#define _RECV_H_

#include <zmq.hpp>
#include <string>
#include <opencv/cv.h>
#include <cppkafka/consumer.h>

struct ImageData {
    cv::Mat img;
    std::string device_id;
    std::string file_name;
};

class ImageSource {
public:
    ImageSource(const char* address, bool file_mode);
    ~ImageSource();
    cv::Mat recv();

private:
    zmq::context_t ctx;
    zmq::socket_t* socket;
    zmq::message_t buf;

    int32_t id;
    std::vector<std::string> file_names;
};

class ImageSourceKafka {
public:
    ImageSourceKafka(std::string brokers, std::string group_name, std::string topic_name, std::string fs_prefix);
    ~ImageSourceKafka();
    ImageData recv();

private:
    cppkafka::Consumer *consumer;
    std::string topic_name;
    std::string group_name;
    std::string fs_prefix;
};

#endif
