FROM python
RUN pip install pika
RUN apt-get install curl
RUN curl https://bazel.build/bazel-release.pub.gpg | apt-key add -
RUN echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
RUN apt-get update && apt-get install bazel

COPY ./new_task.py /new_task.py

CMD  python new_task.py
