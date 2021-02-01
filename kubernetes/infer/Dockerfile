FROM python:3.7
RUN pip install pika
RUN apt-get install curl
RUN curl https://bazel.build/bazel-release.pub.gpg | apt-key add -
RUN echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
RUN apt-get update && apt-get install bazel

COPY ./worker_infer.py /worker_infer.py
COPY [Path to the bert repository] bert-annotator
