#ifndef _API_H_
#define _API_H_

#include "def.h"
#include <inttypes.h>
#include "nvdsinfer.h"

typedef void* hd_activity_detector_t;
typedef void* hd_people_detector_t;
typedef void* cv_mat_ptr_t;
typedef const char* pstr_t;

typedef struct activity_detector_config_t {
    char* model_path;
    char* names_path;
    int32_t batch_size;
    float ext_scale;
} activity_detector_config_t;

typedef struct people_detector_config_t {
    char* model_path;
    int32_t batch_size;
    float cls_thres;
    float nms_thres;
} people_detector_config_t;

typedef struct context_t {
    cv_mat_ptr_t img;
    NvDsInferObjectDetectionInfo boxes[MAX_OBJ_CNT];
    int32_t num_boxes;
    pstr_t activities[MAX_OBJ_CNT][NUM_ACTIVITIES];
} context_t;

#ifdef __cplusplus
extern "C"{
#endif

struct context_t* new_context();

int32_t free_ctx(context_t* c);
int32_t free_adc(activity_detector_config_t c);
int32_t free_pdc(people_detector_config_t c);

hd_activity_detector_t new_activity_detector(activity_detector_config_t c);
hd_people_detector_t new_people_detector(people_detector_config_t c);

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
