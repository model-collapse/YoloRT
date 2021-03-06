#ifndef _IMG_H_
#define _IMG_H_

#include <opencv2/opencv.hpp>

inline void mat_8u3c_to_darknet_blob(cv::Mat img, int32_t tensor_height, int32_t tensor_width, int32_t tensor_depth, float* dst) {
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(tensor_width, tensor_height), 0, 0, CV_INTER_CUBIC);

    float* p = dst;
    for (int32_t c = 0; c < tensor_depth; c++) {
        for (int32_t y = 0; y < tensor_height; y++) {
            for (int32_t x = 0; x < tensor_width; x++) {
                *p = ((float)resized.at<cv::Vec3b>(y, x)[tensor_depth - c]);
                p++;
            }
        }
    }
    
    return;
}

#endif

