#include "pub.h"
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

KafkaPublisher::KafkaPublisher(const char* address, const char* topic_name) {
    Configuration config = {
        { "metadata.broker.list", address }
    };
    
    this->producer = new cppkafka::Producer(config);
    this->topic_name = topic_name;
}

KafkaPublisher::~KafkaPublisher() {
}

void KafkaPublisher::publish(std::string device_id, std::string file_name, std::vector<LabeledPeople> people) {
    rapidjson::Document d;
    Document::AllocatorType& allocator = d.GetAllocator();
    rapidjson::Value vdn;
    rapidjson::Value vfn;
    vdn.SetString(device_id);
    vfn.SetString(file_name);
    d.AddMember("device_id", vdn, allocator);
    d.AddMember("file_name", vfn, allocator);

    rapidjson::Value boxes(rapidjson::kArrayType);
    for (auto p : people) {
        rapidjson::Value box(rapidjson::kObjectType);
        rapidjson::Value activities(rapidjson::kObjectType);
        for (auto act : p.activities) {
            activities.PushBack(act, allocator);
        }
        
        box.AddMember("activities", activities, allocator);

        rapidjson::Value loc(rapidjson::kObjectType);
        loc.PushBack(p.loc.left, allocator);
        loc.PushBack(p.loc.top, allocator);
        loc.PushBack(p.loc.width, allocator);
        loc.PushBack(p.loc.height, allocator);

        box.AddMember("loc", loc, allocator);

        boxes.PushBack(box, allocator);
    }

    d.AddMember("boxes", boxes, allocator);

    StringBuffer buffer;
    Writer<StringBuffer> writer(buffer);
    d.Accept(writer);

    std::cerr << "json = " << buffer.GetString() << std::endl;

    cppkafka::MessageBuilder builder(this->topic_name);
    builder.payload(buffer.GetString());
    this->producer->produce(builder);
    this->producer->flush();
}