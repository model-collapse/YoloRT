package main

import (
	"strings"

	"gopkg.in/confluentinc/confluent-kafka-go.v1/kafka"
)

type ResultPublisher struct {
	KafkaBrokers []string

	topic string
	producer kafka.Producer
	pChan    chan *kafka.Message
}

func NewResultPublisher(brokers []string, topic string) ResultPublisher {
	producer := kafka.NewProducer(&kafka.ConfigMap{
		"metadata.broker.list": strings.Join(borkers, ","),
	})

	return ResultPublisher{
		topic: string,
		KafkaBrokers: brokers,
		producer:     producer,
		pChan:        producer.ProduceChannel,
	}
}

func (r *ResultPublisher) Publish(data string) {
	r.pChan <- &kafka.Message {
		TopicPartition: kafka.TopicPartitions{Topic: &r.topic, Partition: kafka.PartitionAny},
		Value: []byte(data),
	}
}

func (r *ResultPublisher) PublishBytes(data []byte]) {
	r.pChan <- &kafka.Message {
		TopicPartition: kafka.TopicPartitions{Topic: &r.topic, Partition: kafka.PartitionAny},
		Value: data,
	}
}

func (r *ResultPublisher) Close() {
	if r.producer != nil {
		r.producer.Close()
	}
}