#include "recv.h"
#include <opencv2/opencv.hpp>
#include <inttypes.h>
#include "dirent.h"
#include <rapidjson/document.h>
#include "restclient-cpp/restclient.h"

#define MAX_BUF_SIZE 1000000 // 1M buffer

bool endWith(const std::string &str, const std::string &tail) {
	return str.compare(str.size() - tail.size(), tail.size(), tail) == 0;
}

std::vector<std::string> list_dir(std::string path, std::string ext) {
    DIR * dir;
    struct dirent * ptr;
    std::vector<std::string> ret;

    dir = opendir((char *)path.c_str()); //打开一个目录
    while((ptr = readdir(dir)) != NULL) //循环读取目录数据
    {
        std::string x = ptr->d_name;
        std::string dir_path = path + std::string("/") + x;
        
        if (endWith(dir_path, ext)) {
            ret.push_back(dir_path);
        }
    }
    closedir(dir);//关闭目录指针

    return ret;
}

ImageSource::ImageSource(const char* address, bool file_mode) 
    : ctx(1), buf(MAX_BUF_SIZE) {
    if (file_mode) {
        this->id = 0;
        this->file_names = list_dir(address, "jpg");
        fprintf(stderr, "#%d files loaded!\n", this->file_names.size());
        fprintf(stderr, "first one is %s\n", this->file_names[0].c_str());
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
        fprintf(stderr, "receiving %s\n", this->file_names[this->id].c_str());
        std::string path = this->file_names[this->id];
        this->id = (this->id + 1) % this->file_names.size();
        fprintf(stderr, "received!\n");
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

ImageSourceKafka::ImageSourceKafka(const char* broker_addr, const char* group_name, const char* topic_name, const char* fs_prefix) {
    cppkafka::Configuration config = {
        { "metadata.broker.list", std::string(broker_addr) },
        { "group.id", group_name },
        { "enable.auto.commit", false }
    };

    this->consumer = new cppkafka::Consumer(config);
    this->fs_prefix = fs_prefix;

    // Print the assigned partitions on assignment
    this->consumer->set_assignment_callback([](const cppkafka::TopicPartitionList& partitions) {
        std::cout << "Got assigned: " << partitions << std::endl;
    });

    // Print the revoked partitions on revocation
    this->consumer->set_revocation_callback([](const cppkafka::TopicPartitionList& partitions) {
        std::cout << "Got revoked: " << partitions << std::endl;
    });

    this->topic_name = topic_name;

    this->consumer->subscribe({this->topic_name});
    this->consumer->set_timeout(std::chrono::milliseconds(30 * 1000));
}

ImageSourceKafka::~ImageSourceKafka() {
    if (this->consumer != NULL) {
        delete this->consumer;
    }
}

ImageData ImageSourceKafka::recv() {
    cppkafka::Message msg = this->consumer->poll();
    std::cerr << "here" << std::endl;
    if (msg) {
        if (msg.get_error()) {
            // Ignore EOF notifications from rdkafka
            if (!msg.is_eof()) {
		        std::cerr << "[+] Received error notification: " << msg.get_error() << std::endl;
                std::cerr.flush();
            } else {
                std::cerr << "EOF" << std::endl;
            }
        } else {
            rapidjson::Document d;
	        const cppkafka::Buffer& b = msg.get_payload();
            d.Parse(std::string(b.begin(), b.end()).c_str());
            std::string device_id = d["device_id"].GetString();
            std::string file_name = d["file_name"].GetString();
            this->consumer->commit(msg);

            std::cerr << "device: " << device_id << "\t" << "file:" << file_name << std::endl;
            std::stringstream spath;
            spath << fs_prefix << "/" << device_id << "/" << file_name;
            std::cerr << "downloading from " << spath.str() << "..." << std::endl;

            RestClient::Response r = RestClient::get(spath.str());
            if (r.code != 200) {
                std::cerr << "ERROR CODE = " << r.code << std::endl;
                std::cerr.flush();
                return {cv::Mat(), "", ""};
            }

            cv::Mat raw_data(1, r.body.size(), CV_8UC1, (char*)r.body.c_str());
            return {cv::imdecode(raw_data, cv::IMREAD_COLOR), device_id, file_name};
        }

    } else {
        fprintf(stderr, "fail to poll message from topic: %s\n", this->topic_name.c_str());
        fflush(stderr);
    }

    return {cv::Mat(), "", ""};
}
