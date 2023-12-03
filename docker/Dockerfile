FROM nvcr.io/nvidia/tensorflow:23.08-tf2-py3
# FROM tensorflow/tensorflow:latest-gpu

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y &&\
    apt-get install sudo git libsm6 libxext6 libxrender-dev libgl1-mesa-dev -y

# add user
ARG uid
ARG gid
ARG user
RUN groupadd --gid ${gid} ${user} && useradd -ms /bin/bash --uid ${uid} --gid ${gid} ${user}
RUN usermod -aG sudo ${user}

# ssh
RUN apt-get update && apt-get upgrade -y && apt-get install openssh-server -y
RUN mkdir /var/run/sshd
RUN sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config

# poetry
USER ${user}
RUN curl -sSL https://install.python-poetry.org | python3 -
USER root
RUN echo "PATH=/home/${user}/.local/bin${PATH:+:${PATH}}" >> /etc/profile

# c++
RUN apt-get update && apt-get upgrade -y &&\
    apt-get install build-essential cmake -y
