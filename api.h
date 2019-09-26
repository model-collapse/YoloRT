#ifndef _API_H_
#define _API_H_
#define MAX_OBJ_CNT 100
#define NUM_ACTIVITIES 10

#include "nvdsinfer.h"

typedef hd_activity_detector_t void*;
typedef hd_people_detector_t void*;
typedef cv_mat_ptr_t void*;

struct activity_detector_config_t {
    char* model_path;
    char* names_path;
    int32_t batch_size;
    float ext_scale;
};

struct people_detector_config_t {
    char* model_path;
    int32_t batch_size;
    float cls_thres;
    float nms_thres;
};

struct context_t {
    cv_mat_ptr_t img;
    NvDsInferObjectDetectionInfo boxes[MAX_OBJ_CNT];
    int32_t num_boxes;
    const char* activities[MAX_OBJ_CNT][NUM_ACTIVITIES];
};

#ifdef __cplusplus
extern "C"{
#endif

context_t* new_context();

int32_t free_ctx(context_t* c);
int32_t free_adc(activity_detector_config_t c);
int32_t free_pdc(people_detector_config_t c);

hd_activity_detector_t new_activity_detector(activity_detector_config_t c);
hd_people_detector_t new_people_detector(activity_detector_config_t c);

int32_t free_ad(hd_activity_detector_t d);
int32_t free_pd(hd_people_detector_t d);

int32_t infer_ad(hd_activity_detector_t ad, context_t* c);
int32_t infer_pd(hd_people_detector_t ad, context_t* c);

cv_mat_ptr_t imdecode(void* buf, int32_t size);
int32_t free_cvmat(cv_mat_ptr_t a);

#ifdef __cplusplus
}
#endif

#endif