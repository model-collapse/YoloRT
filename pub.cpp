#include "pub.h"
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

KafkaPublisher::KafkaPublisher(const char* address, const char* topic_name) {
    cppkafka::Configuration config = {
        { "metadata.broker.list", address }
    };
    
    this->producer = new cppkafka::Producer(config);
    this->topic_name = topic_name;
}

KafkaPublisher::~KafkaPublisher() {
}

void KafkaPublisher::publish(std::string device_id, std::string file_name, std::vector<LabeledPeople> people) {
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
            activities.PushBack(rapidjson::StringRef(act.activity.c_str()), allocator);
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

    std::string json_data = std::string(buffer.GetString());
    std::cerr << "json = " << json_data << std::endl;

    cppkafka::MessageBuilder builder(this->topic_name);
    builder.payload(json_data);
    this->producer->produce(builder);
    this->producer->flush();
}
