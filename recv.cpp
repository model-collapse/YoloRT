#include "recv.h"
#define MAX_BUF_SIZE 1000000 // 1M buffer

ImageSource::ImageSource(const char* address) 
    : ctx(1), buf(MAX_BUF_SIZE) {
    this->socket = zmq::socket_t(this->ctx, ZMQ_SUB);
    this->socket.connect(address);
    this->socket.setsocketopt(zmq.SUBSCRIBE, "", 0);
    int32_t conflate = 1;
    this->socket.setsocketopt(zmq.CONFLATE, &conflate, sizeof(conflate));
}

ImageSource::~ImageSource() {
    this->socket.close();
}

cv::Mat ImageSource::recv() {
    this->socket.recv(&this->buf);
    return cv::imdecode(this->buf.data(), cv::IMREAD_COLOR);
}