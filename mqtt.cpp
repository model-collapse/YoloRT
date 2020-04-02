#include "mqtt.h"
#include <opencv2/opencv.hpp>
#include <inttypes.h>
#include "dirent.h"
#include "mqtt/client.h"
#include "imagepack.ph.h"
#include "mqtt/client.h"

mqtt::connect_options conn_opts;
mqtt::client* client;

int32_t init_mqtt_client(const char* address, const char* client_name, bool clean_session) {
    conn_opts.set_keep_alive_interval(20);
    conn_opts.set_clean_session(clean_session);
    client = new mqtt::client(address, client_name);
    std::cerr << "Start to connect to MQTT @ " << address << std::endl;
    mqtt::connect_response rsp = client.connect(conn_opts);
    std::cerr << "Connected!" << std::endl;
}

int32_t close_mqtt_client() {
    client->close();
    delete client;
}

ImageSourceMQTT::ImageSourceMQTT(const char* topic) 
{
    std::cerr << "Subscribing..." << std::endl;
    client->subscribe(std::vector<string>{std::string(topic)}, std::vector<int>{0});
    std::cerr << "OK" << std::endl;
}

ImageSourceMQTT::~ImageSourceMQTT() {
}

ImageData ImageSourceMQTT::recv() {
    auto msg = client->consume_message()
    ImgPack pack;
    pack.ParseFromString(string(msg.payload, msg.payloadlen));
    
    ImageData ret;
    cv::Mat raw_data(1, pack.image().size(), CV_8UC1, pack.image());
    ret.img = cv::imdecode(raw_data, cv::IMREAD_COLOR);
    ret.device_id = pack.device_id();
    ret.timestamp = pack.time_stamp_send();

    return ret;
}

ResultPublisher::ResultPublisher(const char* topic_name) {
    this->topic = topic_name;
}

ResultPublisher::~ResultPublisher() {
}

void ResultPublisher::publish(const char* desvice_id, int64_t timestamp, std::vector<LabeledPeople> people) {
    auto beg_doc = std::chrono::system_clock::now();
    rapidjson::Document d;
    d.SetObject();
    rapidjson::Document::AllocatorType& allocator = d.GetAllocator();
    rapidjson::Value vdn;
    rapidjson::Value vfn;
    vdn.SetString(device_id.c_str(), device_id.size());
    vfn.SetString(file_name.c_str(), file_name.size());
    d.AddMember("device_id", vdn, allocator);
    d.AddMember("file_name", vfn, allocator);

    rapidjson::Value boxes(rapidjson::kArrayType);
    for (auto p : people) {
        rapidjson::Value box(rapidjson::kObjectType);
        rapidjson::Value activities(rapidjson::kArrayType);
        for (auto act : p.activities) {
            rapidjson::Value av(act.activity.c_str(), act.activity.size(), allocator);
            //av.SetString(act.activity.c_str(), act.activity.size());
            activities.PushBack(av, allocator);
        }
        
        box.AddMember("activities", activities, allocator);

        rapidjson::Value loc(rapidjson::kArrayType);
        loc.PushBack(p.loc.left, allocator);
        loc.PushBack(p.loc.top, allocator);
        loc.PushBack(p.loc.width, allocator);
        loc.PushBack(p.loc.height, allocator);

        box.AddMember("loc", loc, allocator);

        boxes.PushBack(box, allocator);
    }

    d.AddMember("boxes", boxes, allocator);

    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    d.Accept(writer);

    auto end_doc = std::chrono::system_clock::now();

    std::string json_data = std::string(buffer.GetString());
    std::cerr << "json = " << json_data << std::endl;

    auto beg_prod = std::chrono::system_clock::now();
    auto msg = mqtt::make_message(this->topic, json_data);
    msg->set_qos(0);
    client->publish(msg);

    auto msecs = [](std::chrono::system_clock::time_point beg, std::chrono::system_clock::time_point end) -> int {
            return std::chrono::duration_cast<std::chrono::milliseconds>(end - beg).count();
        };

    std::cerr << "[PD time] | json:" << msecs(beg_doc, end_doc) << "ms, prod:" << msecs(beg_prod, end_prod) << "ms" << std::endl;
}