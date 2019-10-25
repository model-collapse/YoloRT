GO:=go

LIBS:= -lnvinfer_plugin -lnvinfer -lnvparsers -L/usr/local/cuda/lib64 -lcudart -lcublas -L_3rdparty/lib -lhashtable -lzookeeper -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -L/usr/lib/aarch64-linux-gnu -lcppkafka -lrestclient-cpp -lcurl -lboost_program_options -lzmq -lstdc++fs -lpthread
LFLAGS:= -g -Wl,--start-group $(LIBS) -Wl,--end-group

all: yolo_detection

yolo_detection:
    $(GO) build --ldflags "$(LFLAGS)" -o yolo_detection
