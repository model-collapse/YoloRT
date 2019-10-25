GO:=go

CFLAGS:= -Wall -fPIC -g -DTHREADED
CFLAGS+= -I/opt/nvidia/deepstream/deepstream-4.0/sources/includes/ -I/usr/local/cuda/include -I/usr/include -I/usr/src/tensorrt/samples/common -I_3rdparty/include
LIBS:= -lnvinfer_plugin -lnvinfer -lnvparsers -L/usr/local/cuda/lib64 -lcudart -lcublas -L../_3rdparty/lib -lhashtable -lzookeeper -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -L/usr/lib/aarch64-linux-gnu -lcppkafka -lrestclient-cpp -lcurl -lboost_program_options -lzmq -lstdc++fs -lpthread -L../ -lyolo_detection
LFLAGS:= -g -Wl,--start-group $(LIBS) -Wl,--end-group

all: yolo_detection

yolo_detection:
	GOPROXY="goproxy.io" GOPATH="$(HOME)/gpath" CGO_CFLAGS="$(CFLAGS)" CGO_LDFLAGS="$(LFLAGS)" $(GO) env
	GOPROXY="goproxy.io" GOPATH="$(HOME)/gpath" CGO_CFLAGS="$(CFLAGS)" CGO_LDFLAGS="$(LFLAGS)" $(GO) build -o yolo_detection
