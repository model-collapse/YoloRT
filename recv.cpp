#include "recv.h"
#include <opencv2/opencv.hpp>
#include <inttypes.h>
#include "dirent.h"
#define MAX_BUF_SIZE 1000000 // 1M buffer

std::vector<std::string> list_dir(std::string path) {
    DIR * dir;
    struct dirent * ptr;
    std::vector<std::string> ret;

    dir = opendir((char *)path.c_str()); //打开一个目录
    while((ptr = readdir(dir)) != NULL) //循环读取目录数据
    {
        std::string x = ptr->d_name;
        std::string dir_path = path + std::string("/") + x;
        ret.push_back(dir_path);
    }
    closedir(dir);//关闭目录指针

    return ret;
}

ImageSource::ImageSource(const char* address, bool file_mode) 
    : ctx(1), buf(MAX_BUF_SIZE) {
    if (file_mode) {
        this->id = 0;
        this->file_names = list_dir(address);
        fprintf(stderr, "#%d files loaded!", this->files_names.size());
        return;
    }

    this->id = -1;

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
    if (this->id >= 0) {
        std::string path = this->file_names[this->id];
        this->id ++;
        return cv::imread(path.c_str(), cv::IMREAD_COLOR);
    }
    
    fprintf(stderr, "receiving\n");
    fflush(stderr);
    this->socket->recv(&this->buf);
    cv::Mat raw_data(1, this->buf.size() - sizeof(int64_t), CV_8UC1, (char*)this->buf.data() + sizeof(int64_t));
    fprintf(stderr, "received at" PRId64 "\n", ((int64_t*)this->buf.data())[0]);
    fflush(stderr);
    return cv::imdecode(raw_data, cv::IMREAD_COLOR);
}
