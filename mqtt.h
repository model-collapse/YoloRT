#ifndef _RECV_H_
#define _RECV_H_

#include <zmq.hpp>
#include <string>
#include <opencv/cv.h>

struct ImageData {
    cv::Mat img;
    std::string device_id;
    int64_t timestamp;
};

class ImageSourceMQTT {
public: 
    ImageSourceMQTT(const char* topic_name);
    ~ImageSourceMQTT();
    ImageData recv();
};

class ResultPublisher {
public:
    ResultPublisher(const char* topic_name);
    ~ResultPublisher();
    void publish(const char* desvice_id, int64_t timestamp, std::vector<LabeledPeople> people);
private:
    std::string topic:
};

int32_t init_mqtt_client(const char* address, const char* client_name, bool clean_session);
int32_t close_mqtt_client();

#endif
