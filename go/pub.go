package main

import (
	"strings"
	"log"
	"gopkg.in/confluentinc/confluent-kafka-go.v1/kafka"
)

type ResultPublisher struct {
	KafkaBrokers []string

	topic string
	producer *kafka.Producer
	pChan    chan *kafka.Message
}

func NewResultPublisher(brokers []string, topic string) ResultPublisher {
	producer, err := kafka.NewProducer(&kafka.ConfigMap{
		"metadata.broker.list": strings.Join(brokers, ","),
	})

	if err != nil {
		log.Fatalf("Fail to create producer, %v", err)
	}

	return ResultPublisher{
		topic: topic,
		KafkaBrokers: brokers,
		producer:     producer,
		pChan:        producer.ProduceChannel(),
	}
}

func (r *ResultPublisher) Publish(data string) {
	r.pChan <- &kafka.Message {
		TopicPartition: kafka.TopicPartition{Topic: &r.topic, Partition: kafka.PartitionAny},
		Value: []byte(data),
	}
}

func (r *ResultPublisher) PublishBytes(data []byte) {
	r.pChan <- &kafka.Message {
		TopicPartition: kafka.TopicPartition{Topic: &r.topic, Partition: kafka.PartitionAny},
		Value: data,
	}
}

func (r *ResultPublisher) Close() {
	if r.producer != nil {
		r.producer.Close()
	}
}
