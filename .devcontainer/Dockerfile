FROM nvidia/cuda:11.7.0-devel-ubuntu20.04

ARG USERNAME="vscode"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && \
    apt-get -y install --no-install-recommends \
        git vim cmake openmpi-bin libopenmpi-dev

RUN useradd -m -s /bin/bash $USERNAME
USER vscode
RUN git config --global user.email "okumurahidekikangeki@gmail.com"
RUN git config --global user.name "okumura.hideki"