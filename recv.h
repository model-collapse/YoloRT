#ifndef _RECV_H_
#define _RECV_H_

#include <zmq.hpp>
#include <opencv/cv.h>

class ImageSource {
public:
    ImageSource(const char* address);
    ~ImageSource();
    cv::Mat recv();

private:
    zmq::context_t ctx;
    zmq::socket_t socket;
    zmq::message_t buf;
};

#endif
