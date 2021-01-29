#!/usr/bin/env python
import pika
import time

credentials = pika.PlainCredentials(username="[USERNAME]",
                                    password="[PASSWORD]")
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host="[EXTERNAL IP]",
                              credentials=credentials,
                              heartbeat=60 * 120))
channel = connection.channel()

channel.queue_declare(queue="task_queue_convert", durable=True)

while True:
    method_frame, header_frame, body = channel.basic_get("task_queue_convert")
    if method_frame:
        print("Got element from queue:", method_frame, header_frame, body)
        itemstr = body.decode()
        print(" [x] Received %s" % itemstr)
        directory = "/commoncrawl_new"
        bashCommand = "cd bert-annotator && bazel build //training:convert_data && TFHUB_CACHE_DIR=gs://research-brain-bert-annotator-xgcp/cache/ ./bazel-bin/training/convert_data  --train_data_input_path  gs://research-brain-bert-annotator-xgcp/data/distill/cluster" + directory + "/input_part_" + itemstr + ".binproto     --train_data_output_path gs://research-brain-bert-annotator-xgcp/data/distill/cluster" + directory + "/input_part_" + itemstr + ".tfrecord"
        import subprocess
        try:
            output = subprocess.check_output(bashCommand,
                                             stderr=subprocess.STDOUT,
                                             shell=True)
            print("Operation successfull. Output: ", output)
        except subprocess.CalledProcessError as e:
            print("Error during execution")
            print("Exception: ", e)
            print("Process output: ", e.output)
            raise
        print(" [x] Done")
        channel.basic_ack(delivery_tag=method_frame.delivery_tag)
    else:
        print("No message returned")
        print("Stopping this node, no new pods will be created")
        exit(0)
