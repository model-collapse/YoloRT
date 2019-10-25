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

func main() {
	if err := LoadConfigure(CFGPath); err != nil {
		log.Fatalf("Fail to load configure, %v", err)
	}

	pCfg := C.people_detector_config_t{
		model_path: C.CString(GCfg.Yolo.ModelFile),
		batch_size: 1,
		cls_thres:  GCfg.Yolo.ClassThreshold,
		nms_thres:  GCfg.Yolo.NMSThreshold,
	}
	peopleDetector := C.new_people_detector(pCfg)
	defer C.free_pd(peopleDetector)

	aCfg := C.activity_detector_config_t{
		model_path: C.CString(GCfg.Act.ModelFile),
		names_path: C.CString(GCfg.Act.NameFile),
		batch_size: GCfg.Act.BatchSize,
		ext_size:   GCfg.Act.ExtSize,
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

	imgChan := make(chan unsafe.Pointer, GCfg.BufSize)
	resChan := make(LabeledPerson, GCfg.BufSize)

	var wg1 sync.WaitGroup
	var wg2 sync.WaitGroup
	var wg3 sync.WaitGroup
	running := true
	wg1.Add(1)
	go func() {
		defer wg1.Done()
		for running {
			img := src.Recv()
			imgChan <- img
		}
	}()

	wg3.Add(1)
	go func() {
		defer wg3.Done()
		for lbl := range resChan {
			data, _ := json.Marshal(lbl)
			pub.PublishBytes(data)
		}
	}()

	wg3.Add(1)
	go func() {
		defer wg2.Done()
		for img := range imgChan {
			if img == nil {
				log.Printf("nil token received, breaking")
				break
			}

			ctx := C.new_context()
			ctx.img = C.cv_mat_ptr_t(img)

			C.infer_pd(peopleDetector, ctx)
			C.infer_ad(activityDetector, ctx)

			lbl := parseDetectionResultFromC(ctx)
			resChan <- lbl
		}
	}()

	chanTerm := make(os.Signal)
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