#!/bin/bash

REGISTRY=localhost:5000
TAG=graph-compiler-runner:0.0.2

kubectl -n docker-registry port-forward svc/docker-registry 5000:5000 &

cd $(dirname "$0")
docker build . --tag $REGISTRY/$TAG
docker push $REGISTRY/$TAG

