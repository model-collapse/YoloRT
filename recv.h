#ifndef _RECV_H_
#define _RECV_H_

#include <zmq.hpp>
#include <opencv/cv.h>
#include <cppkafka/consumer.h>

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
    ImageSourceKafka(const char* broker_addr, const char* topic_name, const char* fs_prefix);
    ~ImageSourceKafka();
    cv::Mat recv();

private:
    cppkafka::Consumer *consumer;
    std::string topic_name;
    std::string fs_prefix;
}

#endif
