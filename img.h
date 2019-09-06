#ifndef _IMG_H_
#define _IMG_H_

#include <opencv2/opencv.h>

void mat_8u3c_to_darknet_blob(cv::Mat img, int32_t tensor_height, int32_t tensor_width, int32_t tensor_depth, float* dst) {
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(tensor_width, tensor_height), 0, 0, CV_INTER_CUBIC);

    for (int32_t c = 0; c < tensor_depth; c++) {
        for (int32_t y = 0; y < tensor_height; y++) {
            for (int32_t x = 0; x < tensor_width; x++) {
                *p = ((float)img_float.at<cv::Vec3B>(y, x)[tensor_depth - c]) / 255.0;
                p++;
            }
        }
    }
    
    return
}

#endif

