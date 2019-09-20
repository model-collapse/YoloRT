################################################################################
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

CC:= g++
NVCC:=/usr/local/cuda/bin/nvcc

CFLAGS:= -Wall -std=c++11 -fPIC -g -DTHREADED
CFLAGS+= -I/opt/nvidia/deepstream/deepstream-4.0/sources/includes/ -I/usr/local/cuda/include -I/usr/include -I/usr/src/tensorrt/samples/common -I_3rdparty/include

LIBS:= -lnvinfer_plugin -lnvinfer -lnvparsers -L/usr/local/cuda/lib64 -lcudart -lcublas -L_3rdparty/lib -lhashtable -lzookeeper -L/usr/lib -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -L/usr/lib/aarch64-linux-gnu -lcppkafka -lrestclient-cpp -lcurl -lboost_program_options -lzmq -lstdc++fs -lpthread
LFLAGS:= -g -Wl,--start-group $(LIBS) -Wl,--end-group

INCS:= $(wildcard *.h)
BASE_SRCFILES:= recv.cpp \
	   activity_detection.cpp    \
           people_counting.cpp    \
           nvdsparsebbox_Yolo.cpp   \
           nvdsinfer_yolo_engine.cpp \
           yoloPlugins.cpp    \
           trt_utils.cpp              \
           yolo.cpp             \
		   pub.cpp 				\
		   cfg.cpp				\
		   zk.cpp 				\
           kernels.cu       
TARGET_EXEC:= yolo_detection
CVT_EXEC:= convert_to_trt
TRY_EXEC:= trt_plugin_try
OLD_EXEC:=old_yolo_detection

TARGET_OBJS:= $(BASE_SRCFILES:.cpp=.o)
TARGET_OBJS:= $(TARGET_OBJS:.cu=.o)

all: $(TARGET_EXEC)

%.o: %.cpp $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

%.o: %.cu $(INCS) Makefile
	$(NVCC) -c -o $@ --compiler-options '-fPIC' $<

$(TARGET_EXEC) : $(TARGET_OBJS) main_comp.o
	$(CC) -o $@  main_comp.o $(TARGET_OBJS) $(LFLAGS)

$(OLD_EXEC) : $(TARGET_OBJS) main_comp_s.o
	$(CC) -o $@  main_comp_s.o $(TARGET_OBJS) $(LFLAGS)

$(CVT_EXEC) : $(TARGET_OBJS) cvt.o
	$(CC) -o $@  cvt.o $(TARGET_OBJS) $(LFLAGS)

$(TRY_EXEC) : plugin_try.o
	$(CC) -o $@  plugin_try.o $(LFLAGS)



clean:
	rm -rf $(TARGET_EXEC)
