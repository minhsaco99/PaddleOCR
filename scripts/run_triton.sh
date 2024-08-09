#!/bin/sh

http_port=11003
grpc_port=11004
metrics_port=11005

docker run --gpus=0 -p $http_port:$http_port -p $grpc_port:$grpc_port -p $metrics_port:$metrics_port --net=host \
            -v $PWD/models/triton:/models nvcr.io/nvidia/tritonserver:22.07-py3 tritonserver \
            --model-repository=/models --strict-model-config=true \
            --http-port $http_port --grpc-port $grpc_port --metrics-port $metrics_port