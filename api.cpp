#include "api.h"
#include "nvdsinfer_custom_impl.h"
#include "common.h"
#include <opencv2/opencv.hpp>
#include "people_counting.h"
#include "activity_detection.h"

Logger gLogger;

void safe_free(void* c) {
    if (c != NULL) {
        free(c);
    }
}

context_t* new_context() {
    context_t* ret = (context_t*) malloc(sizeof(context_t));
    ret->img = NULL;
    memset(ret->boxes, 0, sizeof(NvDsInferObjectDetectionInfo) * MAX_OBJ_CNT);
    ret->num_boxes = MAX_OBJ_CNT;
    memset(ret->activities, 0, sizeof(char**) * MAX_OBJ_CNT * NUM_ACTIVITIES);
    return ret;
}

int32_t free_ctx(context_t* c) {
    safe_free(c);
    return 0;
}

int32_t free_adc(activity_detector_config_t c) {
    safe_free(c.model_path);
    safe_free(c.names_path);
    return 0;
}

int32_t free_pdc(people_detector_config_t c) {
    safe_free(c.model_path);
    return 0;
}

hd_activity_detector_t new_activity_detector(activity_detector_config_t c) {
    return new ActivityDetector(c.model_path, c.names_path, c.batch_size, c.ext_scale, gLogger);
}

hd_people_detector_t new_people_detector(people_detector_config_t c) {
    return new PeopleDetector(c.model_path, c.batch_size, c.cls_thres, c.nms_thres, gLogger);
}

int32_t free_ad(hd_activity_detector_t d) {
    ActivityDetector* p = (ActivityDetector*)d;
    delete p;
    return 0;
}

int32_t free_pd(hd_people_detector_t d) {
    PeopleDetector* p = (PeopleDetector*)d;
    delete p;
    return 0;
}

cv_mat_ptr_t imdecode(void* buf, int32_t size) {
    cv::Mat raw_data(1, size, CV_8UC1, (char*)buf);
    cv::Mat* ret = new cv::Mat();
    cv::imdecode(raw_data, cv::IMREAD_COLOR, ret);
    return ret;
}

int32_t free_cvmat(cv_mat_ptr_t a) {
    safe_free(a);
    return 0;
}

int32_t infer_ad(hd_activity_detector_t ad, context_t* c) {
    cv::Mat* img = (cv::Mat*)ad;
    ActivityDetector* det = (ActivityDetector*)ad;
    return det->detect_capi(*img, c->boxes, c->num_boxes, c->activities);
}

int32_t infer_pd(hd_people_detector_t pd, context_t* c) {
    cv::Mat* img = (cv::Mat*)pd;
    PeopleDetector* det = (PeopleDetector*)pd;
    return det->detect_capi(*img, c->boxes, c->num_boxes);
}
