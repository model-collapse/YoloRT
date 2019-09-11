#include <stdio.h>
#include <stdint.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "recv.h"
#include "people_counting.h"
#include "activity_detection.h"

const char* PC_CFG_PATH = "../../../model/darknet/yolov3_person.cfg";
const char* PC_WTS_PATH = "../../../model/darknet/yolov3_person_16000.weights";
const int32_t PC_BATCH_SIZE = 1;

const char* AD_CFG_PATH = "../../../model/darknet/wwdarknet53v2.cfg";
const char* AD_WTS_PATH = "../../../model/darknet/wwdarknet53v2_50000.weights";
const char* AD_NAME_PATH = "../../../model/darknet/activity_wework.names";
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

    char text_buf[MAX_TEXT_LEN];
    sprintf(text_buf, "%s:%03f", person.activity.c_str(), person.prob);
    std::string text(text_buf);

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

    ActivityDetector ad(AD_CFG_PATH, AD_WTS_PATH, AD_NAME_PATH, ACT_DET_BATCH_SIZE, gLogger);
    PeopleDetector pd(PC_CFG_PATH, PC_WTS_PATH, PC_BATCH_SIZE, gLogger);

    // creating image source
    ImageSource src("tcp://10.249.77.88:18964");

    int32_t frames = 0;
    while (true) {
        frames ++;
        auto img = src.recv();
        
        auto boxes = pd.detect(img);
        std::cerr << "[people count] " << boxes.size() << " were found" << std::endl;

        auto persons = ad.detect(img, boxes);
        std::cerr << "[marked]" << std::endl;

        cv::Mat canvas = img.clone();
        for (auto person : persons) {
            mark_a_labeled_person(canvas, person);
        }

        char pathBuf[30];
        sprintf(pathBuf, "dump/frame_%d.jpg", frames);
        cv::imwrite(pathBuf, canvas);
        sprintf(pathBuf, "dump/orig_%d.jpg", frames);
        cv::imwrite(pathBuf, img);

        if (frames >= 500) {
            break;
        }
    }
}
