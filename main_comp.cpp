#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "recv.h"
#include "pub.h"
#include "people_counting.h"
#include "activity_detection.h"

const char* PC_MODEL_PATH = "../../../model/tensorRT/yolov3_person.trt.dat";
const int32_t PC_BATCH_SIZE = 1;
const char* AD_MODEL_PATH = "../../../model/tensorRT/wwdarknet53v2.trt.dat";
const char* AD_NAME_PATH = "../../../model/darknet/activity_wework.names.with_thres";
const int32_t ACT_DET_BATCH_SIZE = 4;

const int32_t MAX_TEXT_LEN = 20;

Logger gLogger;

void mark_a_people(cv::Mat canvas, NvDsInferObjectDetectionInfo people) {
    const static cv::Scalar color(0, 255, 255);
    cv::rectangle(canvas, cv::Point(people.left, people.top), cv::Point(people.left + people.width, people.top + people.height), color, 2);
}

void mark_a_labeled_person(cv::Mat canvas, LabeledPeople person) {
    mark_a_people(canvas, person.loc);

    auto font = cv::FONT_HERSHEY_SIMPLEX;
    float font_scale = 0.5;
    float thickness = 1;

    std::stringstream ss;
    for (Activity act : person.activities) {
        ss << act.activity << ":";
    }

    std::string text = ss.str();

    int32_t anchor;
    cv::Size text_size = cv::getTextSize(text, font, font_scale, thickness, &anchor);
    
    const int32_t margin = 3;

    int32_t text_top = person.loc.top;
    if (text_top < 0) {
        return;
    }
    int32_t text_left = person.loc.left;
    int32_t text_height = anchor + text_size.height + margin * 2;
    int32_t text_width = text_size.width + margin * 2;

    const static cv::Scalar color(0, 255, 255);
    const static cv::Scalar txt_color(0, 0, 0);
    cv::rectangle(canvas, cv::Rect(text_left, text_top, text_width, text_height), color, CV_FILLED);

    cv::putText(canvas, text, cv::Point(text_left + margin, text_top + text_height - margin - anchor), font, font_scale, txt_color, thickness);
}

int32_t main(int32_t argc, char** argv) {
    fprintf(stderr, "haha\n");
    fflush(stderr);

    // creating image source
    ImageSourceKafka src("10.249.77.87:9092", "2" , "model_commands", "http://10.249.77.82:8000");
    KafkaPublisher pub("10.249.77.87:9092", "model_results");

    ActivityDetector ad(AD_MODEL_PATH, AD_NAME_PATH, ACT_DET_BATCH_SIZE, gLogger);
    PeopleDetector pd(PC_MODEL_PATH, PC_BATCH_SIZE, gLogger);

    int32_t frames = 0;
    while (true) {
        frames ++;
        auto img_data = src.recv();
        if (img_data.img.cols == 0 || img_data.img.rows == 0) {
            std::cerr << "empty image" << std::endl;
            continue;
        }

        auto boxes = pd.detect(img_data.img);
        std::cerr << "[people count] " << boxes.size() << " were found" << std::endl;

        auto persons = ad.detect(img_data.img, boxes);
        std::cerr << "[marked]" << std::endl;

       	pub.publish(img_data.device_id, img_data.file_name, persons); 
    }
}
