#!/bin/bash

python -m grpc_tools.protoc -I ./proto --python_out=patter/server --grpc_python_out=patter/server ./proto/speech.proto
