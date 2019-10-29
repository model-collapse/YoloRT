package main

// #include "api.h"
import "C"

import (
	"log"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"unsafe"

	json "github.com/json-iterator/go"
)

const (
	CFGPath = "cfg.json"
)

type Image struct {
	DeviceName string
	FileName   string
	Err error
	Image      unsafe.Pointer
}

type PublishResult struct {
	DeviceName string `json: "device_id"`
	FileName   string `json: "file_name"`
	Boxes []LabeledPerson `json:"boxes"`
}

func main() {
	if err := LoadConfigure(CFGPath); err != nil {
		log.Fatalf("Fail to load configure, %v", err)
	}

	pCfg := C.people_detector_config_t{
		model_path: C.CString(GCfg.Yolo.ModelFile),
		batch_size: 1,
		cls_thres:  C.float(GCfg.Yolo.ClassThreshold),
		nms_thres:  C.float(GCfg.Yolo.NMSThreshold),
	}
	peopleDetector := C.new_people_detector(pCfg)
	defer C.free_pd(peopleDetector)

	aCfg := C.activity_detector_config_t{
		model_path: C.CString(GCfg.Act.ModelFile),
		names_path: C.CString(GCfg.Act.NameFile),
		batch_size: C.int32_t(GCfg.Act.BatchSize),
		ext_scale:  C.float(GCfg.Act.ExtScale),
	}
	activityDetector := C.new_activity_detector(aCfg)
	defer C.free_ad(activityDetector)

	src := NewImageSource(GCfg.KafkaSub.Brokers,
		parseOffsetMode(GCfg.KafkaSub.OffsetMode),
		GCfg.KafkaSub.TopicName, GCfg.KafkaSub.GroupName,
		GCfg.KafkaSub.Timeout)
	defer src.Close()

	pub := NewResultPublisher(GCfg.KafkaPub.Brokers, GCfg.KafkaPub.TopicName)
	defer pub.Close()

	imgChan := make(chan Image, GCfg.BufSize)
	resChan := make(chan PublishResult, GCfg.BufSize)

	var wg1 sync.WaitGroup
	var wg2 sync.WaitGroup
	var wg3 sync.WaitGroup
	running := true
	wg1.Add(1)
	go func() {
		defer wg1.Done()
		for running {
			img := src.Recv()
			if img.Err != nil {
				continue
			}

			imgChan <- img
		}
	}()

	wg3.Add(1)
	go func() {
		defer wg3.Done()
		for res := range resChan {
			data, _ := json.Marshal(res)
			pub.PublishBytes(data)
		}
	}()

	wg3.Add(1)
	go func() {
		defer wg2.Done()
		for img := range imgChan {
			if img.Err != nil {
				log.Printf("Error in image, skip")
				continue
			}

			ctx := C.new_context()
			ctx.img = C.cv_mat_ptr_t(img.Image)

			C.infer_pd(peopleDetector, ctx)
			C.infer_ad(activityDetector, ctx)

			lbl := parseDetectionResultFromC(ctx)
			resChan <- PublishResult {
				FileName: img.FileName,
				DeviceName: img.DeviceName,
				Boxes: lbl,
			}
		}
	}()

	chanTerm := make(chan os.Signal)
	signal.Notify(chanTerm, syscall.SIGTERM)

	<-chanTerm

	log.Printf("exiting...")
	running = false
	wg1.Wait()
	close(imgChan)
	wg2.Wait()
	close(resChan)
	wg3.Wait()
}
