FROM ubuntu:latest
LABEL authors="kelwar"

ENTRYPOINT ["top", "-b"]