FROM golang:1.20.5-bullseye AS build
ARG LIBTENSORFLOW_FILENAME=libtensorflow-cpu-linux-x86_64-2.12.0.tar.gz
ARG LIBTENSORFLOW_URL=https://storage.googleapis.com/tensorflow/libtensorflow/${LIBTENSORFLOW_FILENAME}

WORKDIR /app

RUN apt update; \
    apt install -y ffmpeg ca-certificates upx

RUN wget ${LIBTENSORFLOW_URL}; \
    tar -C /usr/local -xzf ${LIBTENSORFLOW_FILENAME}

ENV LD_LIBRARY_PATH=/usr/local/lib

COPY . .

RUN go build -ldflags="-s -w" -o /app_bin.fat
RUN upx --lzma -o /app_bin /app_bin.fat


FROM debian:bullseye-slim
ARG LIBTENSORFLOW_FILENAME=libtensorflow-cpu-linux-x86_64-2.12.0.tar.gz
ARG MODEL_FOLDER=mobilenet_v2_140_224
WORKDIR /app

RUN apt update; \
    apt install -y ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /app/${MODEL_FOLDER} ${MODEL_FOLDER}
COPY --from=build /app/${LIBTENSORFLOW_FILENAME} ${LIBTENSORFLOW_FILENAME}
COPY --from=build /app_bin app_bin

RUN tar -C /usr/local -xzf ${LIBTENSORFLOW_FILENAME} \
    && rm ${LIBTENSORFLOW_FILENAME}

ENV LD_LIBRARY_PATH=/usr/local/lib

EXPOSE 8080
ENTRYPOINT ["/app/app_bin"]
