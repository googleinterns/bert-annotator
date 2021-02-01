#!/usr/bin/env python
import pika
import sys

credentials = pika.PlainCredentials(username="[USERNAME]",
                                    password="[PASSWORD]")
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host="[EXTERNAL IP]", credentials=credentials))
channel = connection.channel()

# queue = "task_queue_infer"
queue = "task_queue_convert"
channel.queue_declare(queue=queue, durable=True)
while True:
    method_frame, header_frame, body = channel.basic_get(queue)
    if method_frame:
        print("removing entry " + body.decode())
        channel.basic_ack(delivery_tag=method_frame.delivery_tag)
    else:
        break

for i in range(0, 277 + 1):
    message = str(i)
    channel.basic_publish(
        exchange="",
        routing_key=queue,
        body=message,
        properties=pika.BasicProperties(
            delivery_mode=2,  # make message persistent
        ))
connection.close()
