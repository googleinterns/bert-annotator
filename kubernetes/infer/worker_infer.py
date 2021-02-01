#!/usr/bin/env python
import pika
import time
import sys
import subprocess

credentials = pika.PlainCredentials(username="[USERNAME]",
                                    password="[PASSWORD]")
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host="[EXTERNAL IP]",
                              credentials=credentials,
                              heartbeat=60 * 30))
channel = connection.channel()

channel.queue_declare(queue="task_queue_infer", durable=True)
#print(" [*] Waiting for messages. To exit press CTRL+C")

#def callback(ch, method, properties, body):

while True:
    method_frame, header_frame, body = channel.basic_get("task_queue_infer")
    if method_frame:
        print("Got element from queue:", method_frame, header_frame, body)
        itemstr = body.decode()
        print(" [x] Received %s" % itemstr)
        tpu = sys.argv[1]
        print("Using TPU with address", tpu)
        model = "gs://research-brain-bert-annotator-xgcp/training_checkpoints/lucidsky_9000k_tiny/model_170"
        size = "tiny"
        directory = "/commoncrawl"
        bashCommand = "cd bert-annotator && bazel build //training:evaluate && TFHUB_CACHE_DIR=gs://research-brain-bert-annotator-xgcp/cache/ ./bazel-bin/training/evaluate       --size  " + size + "       --model_path " + model + "      --input_paths gs://research-brain-bert-annotator-xgcp/data/distill/cluster" + directory + "/input_" + itemstr + ".tfrecord      --raw_paths gs://research-brain-bert-annotator-xgcp/data/distill/cluster" + directory + "/input_" + itemstr + ".binproto      --output_directory gs://research-brain-bert-annotator-xgcp/data/distill/cluster" + directory + "/gen_9m      --save_output_formats tfrecord --save_output_formats lftxt --save_output_formats binproto   --tpu_address " + tpu + " --batch_size 8 --no_eval"
        try:
            output = subprocess.check_output(bashCommand,
                                             stderr=subprocess.STDOUT,
                                             shell=True)
            print("Operation successfull. Output: ", output)
            print("sys.argv: ", sys.argv)
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
