FROM golang:1.20-bullseye AS build
ARG LIBTENSORFLOW_FILENAME=libtensorflow-cpu-linux-x86_64-2.15.0.tar.gz
ARG LIBTENSORFLOW_URL=https://storage.googleapis.com/tensorflow/libtensorflow/${LIBTENSORFLOW_FILENAME}

WORKDIR /app

RUN wget ${LIBTENSORFLOW_URL}; \
    tar -C /usr/local -xzf ${LIBTENSORFLOW_FILENAME}

ENV LD_LIBRARY_PATH=/usr/local/lib

COPY . .

RUN go build -ldflags="-s -w" -o /app_bin

FROM debian:bullseye-slim
ARG LIBTENSORFLOW_FILENAME=libtensorflow-cpu-linux-x86_64-2.15.0.tar.gz
# ARG MODEL_FOLDER=mobilenet_v2_140_224
ARG MODEL_FOLDER_2=inception_v3

WORKDIR /app

COPY --from=build /app/${MODEL_FOLDER_2} ${MODEL_FOLDER_2}
# COPY --from=build /app/${MODEL_FOLDER} ${MODEL_FOLDER}
COPY --from=build /app/${LIBTENSORFLOW_FILENAME} ${LIBTENSORFLOW_FILENAME}
COPY --from=build /app_bin app_bin

RUN tar -C /usr/local -xzf ${LIBTENSORFLOW_FILENAME} \
    && rm ${LIBTENSORFLOW_FILENAME}

ENV LD_LIBRARY_PATH=/usr/local/lib

EXPOSE 8080
ENTRYPOINT ["/app/app_bin"]
