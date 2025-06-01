#!/bin/sh
DOCKER_TAG=${1:-toadtrade}
docker build --platform linux/amd64 -t $DOCKER_TAG .