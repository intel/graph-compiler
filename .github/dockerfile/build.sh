#!/bin/bash

REGISTRY=localhost:5000
TAG=graph-compiler-runner:latest

cd $(dirname "$0")
docker build . --tag $REGISTRY/$TAG
docker push $REGISTRY/$TAG

