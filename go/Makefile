GO:=go

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
current_dir := $(dir $(mkfile_path))
CFLAGS:= -Wall -fPIC -g -DTHREADED
CFLAGS+= -I/opt/nvidia/deepstream/deepstream-4.0/sources/includes/ -I/usr/local/cuda/include -I/usr/include -I/usr/src/tensorrt/samples/common -I$(current_dir)/../_3rdparty/include -I$(current_dir)/../
LIBS:= -L$(current_dir)/../ -lyolo_detection -lnvinfer_plugin -lnvinfer -lnvparsers -L/usr/local/cuda/lib64 -lcudart -lcublas -L$(current_dir)/../_3rdparty/lib -lhashtable -lzookeeper -L/usr/local/lib -lrdkafka -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -L/usr/lib/aarch64-linux-gnu -lrestclient-cpp -lcurl -lboost_program_options -lzmq -lstdc++fs -lpthread
LFLAGS:= -g -Wl,--start-group $(LIBS) -Wl,--end-group

all: yolo_detection

yolo_detection:
	GOPROXY="goproxy.io" GOPATH="$(HOME)/gpath" CGO_CFLAGS="$(CFLAGS)" CGO_LDFLAGS="$(LFLAGS)" $(GO) env
	GOPROXY="goproxy.io" GOPATH="$(HOME)/gpath" CGO_CFLAGS="$(CFLAGS)" CGO_LDFLAGS="$(LFLAGS)" $(GO) build -o yolo_detection
