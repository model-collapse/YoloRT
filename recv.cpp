#include "recv.h"
#include <opencv2/opencv.hpp>
#define MAX_BUF_SIZE 1000000 // 1M buffer

ImageSource::ImageSource(const char* address) 
    : ctx(1), socket(ctx, ZMQ_SUB), buf(MAX_BUF_SIZE) {
    this->socket.connect(address);
    this->socket.setsockopt(ZMQ_SUBSCRIBE, "", 0);
    int32_t conflate = 1;
    this->socket.setsockopt(ZMQ_CONFLATE, &conflate, sizeof(conflate));
}

ImageSource::~ImageSource() {
    this->socket.close();
}

cv::Mat ImageSource::recv() {
    this->socket.recv(&this->buf);
    cv::Mat raw_data(1, this->buf.size(), CV_8UC1, this->buf.data());
    return cv::imdecode(raw_data, cv::IMREAD_COLOR);
}
