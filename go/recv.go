package main

// #include "api.h"
// #include "stdlib.h"
import "C"

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"encoding/json"
	"log"
	"strings"
	"unsafe"
	"time"
	"gopkg.in/confluentinc/confluent-kafka-go.v1/kafka"
)

const (
	Last = iota
	Full
	Latest
)

func parseOffsetMode(m string) int {
	if m == "Latest" || m == "latest" {
		return Latest
	} else if m == "Full" || m == "full" {
		return Full
	} else {
		return Last
	}
}

type ImageSource struct {
	KafkaBrokers []string
	OffsetMode   int
	TopicName    string
	GroupName    string
	Timeout      int

	consumer   *kafka.Consumer
	partitions []kafka.TopicPartition
}

func NewImageSource(brokers []string, offMode int, topicName, groupName string, timeout int) ImageSource {
	consumer, err := kafka.NewConsumer(&kafka.ConfigMap{
		"metadata.broker.list": strings.Join(brokers, ","),
		"group.id":             groupName,
	})

	if err != nil {
		log.Fatalf("Fail to create consumer: %v", err)
	}

	consumer.SubscribeTopics([]string{topicName}, nil)
	partitions, err := consumer.Assignment()
	if err != nil {
		log.Printf("error in getting assignment: %v", err)
	}

	if offMode == Full {
		for i := range partitions {
			partitions[i].Offset = kafka.OffsetBeginning
			consumer.Seek(partitions[i], timeout)
		}
	}

	return ImageSource{
		KafkaBrokers: brokers,
		OffsetMode:   offMode,
		TopicName:    topicName,
		GroupName:    groupName,
		Timeout:      timeout,
		consumer:     consumer,
		partitions:   partitions,
	}
}

func (s *ImageSource) Recv() Image {
	if s.OffsetMode == Latest {
		for i := range s.partitions {
			s.partitions[i].Offset = kafka.OffsetEnd
			s.consumer.Seek(s.partitions[i], s.Timeout)
		}
	}

	msg, err := s.consumer.ReadMessage(time.Duration(s.Timeout) * time.Millisecond)
	if err != nil {
		log.Printf("error in receiving message, %v", err)
		return Image {
			Err: err,
		}
	}

	cmd := struct {
		DeviceName string `json: "device_id"`
		FileName   string `json: "file_name"`
	}{}

	if err := json.Unmarshal(msg.Value, &cmd); err != nil {
		log.Printf("Error format of message, %v", err)
		return Image {
			Err: err,
		}
	}

	url := fmt.Sprintf("%s/%s/%s", GCfg.FileServer.Addr, cmd.DeviceName, cmd.FileName)
	resp, err := http.Get(url)
	if err != nil {
		log.Printf("Error in getting image from %s", url)
		return Image {
			Err: err,
		}
	}

	defer resp.Body.Close()
	data, _ := ioutil.ReadAll(resp.Body)

	size := len(data)
	cdata := C.CBytes(data)
	ret := C.imdecode(cdata, C.int(size))
	C.free(cdata)

	if s.OffsetMode == Last {
		s.consumer.CommitMessage(msg)
	}

	return Image {
		DeviceName: cmd.DeviceName,
		FileName: cmd.FileName,
		Image: unsafe.Pointer(ret),
	}
}

func (s *ImageSource) Close() {
	if s.consumer != nil {
		s.consumer.Close()
	}
}
