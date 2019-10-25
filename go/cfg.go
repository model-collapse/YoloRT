package main

import (
	"io/ioutil"

	json "github.com/json-iterator/go"
)

type ZKCfg struct {
	Addr      []string `json:"addr"`
	KafkaPath string   `json:"kafka_path"`
}

type YOLOCfg struct {
	ModelFile      string  `json:"model_file"`
	ClassThreshold float32 `json:"class_thres"`
	NMSThreshold   float32 `json:"nms_thres"`
}

type ActCfg struct {
	ModelFile string `json:"model_file"`
	NameFile  string `json:"name_file"`
	BatchSize int32  `json:"batch_size"`
	ExtSize   int32  `json:"ext_size"`
}

type KafkaCfg struct {
	Brokers    []string `json:"brokers"`
	TopicName  string   `json:"topic_name"`
	GroupName  string   `json:"group_name"`
	OffsetMode string   `json:"offset_mode"`
	Timeout    int      `json:"timeout"`
}

type FileServerCfg struct {
	Addr string `json:"addr"`
}

type Configure struct {
	Zookeeper  ZKCfg         `json:"zookeeper"`
	Yolo       YOLOCfg       `json:"yolo"`
	Act        ActCfg        `json:"act"`
	KafkaSub   KafkaCfg      `json:"kafka_sub"`
	KafkaPub   KafkaCfg      `json:"kafka_pub"`
	FileServer FileServerCfg `json:"file_server"`
	BufSize    int           `json"buf_size"`
}

var GCfg Configure

func LoadConfigure(path string) (reterr error) {
	data, reterr := ioutil.ReadFile(path)
	if reterr != nil {
		return
	}

	if reterr = json.Unmarshal(data, &GCfg); reterr != nil {
		return
	}

	return
}
