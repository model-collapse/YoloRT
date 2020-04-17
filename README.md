# YoloRT
The project deploys the yolov3 model on TensorRT for an application of people analytics. A Inception-based activity detection model is also included. This deployment is also optimized for embedded artifical intellegence platform, especially the Jetson Seroes ---- Nvidia targra-based GPU computation unit.

--- to do [image] ---

## Model conversion
Till the end of 2019, there is still no stable version of YoloV3 model other than Darknet. The model used here were trained using Darknet[(link)](https://pjreddie.com/darknet/). The project implements a translation tool which converts the model file directly from Darket format into tensorRT's TRT format. Most of the code was inpired by the Nvidia DeepStream[link](https://developer.nvidia.com/deepstream-sdk) which includes a implementation of YoloV2.  
Just run 
```
make convert_to_trt
./convert_to_trt
```
to do the conversion. Note that there's no commandline arguments here, please change the input/output file path in the cvt.cpp.

## Model deployment
The main program can be built and run with  
```
make yolo_detection
./yolo_detection
```
All the configurations are given in a configuration file named cfg.ini locatd in the same path of the executable. Here is a example of cfg.ini
```
[MQTT]
address=your-mqtt-broker-ip
client_name=name-your-mqtt-client
clean_session=clean-session-flag[true or false]
in_topic=the-topic-to-accquire-image
out_topic=the-topic-to-output-results

[YoloCFG]
model_file=yolo-detection-file
class_thres=0.15
nms_thres=0.45

[ActCFG]
model_file=activity-detection-model-file
name_file=name-list-file-of-activities
batch_size=4
ext_scale=0.4
```

Welcome to post any issue or question and PRs as well.
Contact charlie.yang@outlook.com for more info.