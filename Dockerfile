FROM golang:1.20.5-bullseye

WORKDIR /app

RUN apt-get update; \
    apt-get install -y python3 python3-pip; \
    pip install tensorflow

RUN wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.6.0.tar.gz; \
    tar -C /usr/local -xzf libtensorflow-cpu-linux-x86_64-2.6.0.tar.gz

ENV LD_LIBRARY_PATH=/usr/local/lib

COPY . .

RUN go build -o /app_bin

EXPOSE 8080
ENTRYPOINT ["/app_bin"]