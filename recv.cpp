#include "recv.h"
#include <opencv2/opencv.hpp>
#define MAX_BUF_SIZE 1000000 // 1M buffer

ImageSource::ImageSource(const char* address) 
    : ctx(1), buf(MAX_BUF_SIZE) {
    try{
        this->socket = new zmq::socket_t(this->ctx, ZMQ_SUB);
    } catch (zmq::error_t err) {
        fprintf(stderr, "error throwed\n");
        exit(1);
    }

    fprintf(stderr, "connecting\n");
    fflush(stderr);
    this->socket->connect(address);
    this->socket->setsockopt(ZMQ_SUBSCRIBE, "", 0);
    int32_t conflate = 1;
    this->socket->setsockopt(ZMQ_CONFLATE, &conflate, sizeof(conflate));
    fprintf(stderr, "done\n");
    fflush(stderr);
}

ImageSource::~ImageSource() {
    if (NULL != this->socket) {
        this->socket->close();
    }
}

cv::Mat ImageSource::recv() {
    fprintf(stderr, "receiving");
    fflush(stderr);
    this->socket->recv(&this->buf);
    cv::Mat raw_data(1, this->buf.size(), CV_8UC1, this->buf.data());
    fprintf(stderr, "received");
    fflush(stderr);
    return cv::imdecode(raw_data, cv::IMREAD_COLOR);
}
