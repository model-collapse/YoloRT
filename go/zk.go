package main

import (
	"fmt"
	"log"
	"time"

	json "github.com/json-iterator/go"
	"github.com/samuel/go-zookeeper/zk"
)

type ZKAgent struct {
	Servers []string
	Timeout int
	KafkaZKPath string

	conn    *zk.Conn
}

func NewZKAgent(servers []string, timeout int, kafkaZKPath string) ZKAgent {
	conn, evt, err := zk.Connect(servers, time.Duration(timeout)*time.Millisecond)
	if err != nil {
		log.Fatalf("Fail to connect to ZK server")
	}

	go func() {
		for evt := range evt {
			log.Printf("[ERR ZK] %v", evt.Err)
		}
	} ()

	return ZKAgent {
		Servers: servers,
		Timeout: timeout,
		KafkaZKPath: kafkaZKPath,
		conn: conn,
	}
}

type HostInfo struct {
	Host string `json:"host"`
	Port string `json:"port"`
}

func (h *HostInfo) String() string {
	return fmt.Sprintf("%s:%s", h.Host, h.Port)
}

func (a *ZKAgent) FetchKafkaServers() []string {
	children, _, err := a.conn.Children(a.KafkaZKPath)
	if err != nil {
		log.Printf("[ZK]Fail to accquire children of %s", a.KafkaZKPath)
		return nil
	}

	var ret []string
	for _, child := range children {
		data, _, err := a.conn.Get(a.KafkaZKPath + "/" + child)
		if err != nil {
			log.Printf("[ZK]Fail to access path %s", child)
			continue
		}

		var host HostInfo
		if err := json.Unmarshal(data, &host); err != nil {
			log.Printf("[ZK]Fail to unmarshal host info, %v", err)
			continue
		}

		ret = append(ret, host.String())
	}

	return ret
}


