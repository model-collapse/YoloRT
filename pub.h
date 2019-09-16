#include <cppkafka/producer.h>
#include <cppkafka/configuration.h>
#include "activity_detection.h"

class KafkaPublisher {
public:
    KafkaPublisher(const char* address, const char* topic_name);
    ~KafkaPublisher();
    void publish(std::string device_id, std::string file_name, std::vector<LabeledPeople> people);

private:
    cppkafka::Producer *producer;
    std::string topic_name;
};