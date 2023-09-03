# TODO: can this be merged with the main project Dockerfile?

FROM ubuntu:20.04
LABEL Name=tinychain Version=0.0.2

RUN apt-get -y update && apt-get install -y sudo curl

RUN ln -snf /usr/share/zoneinfo/America/New_York /etc/localtime && echo America/New_York > /etc/timezone

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

RUN apt-get install -y gnupg2 ca-certificates apt-utils software-properties-common

RUN apt-get install -y build-essential

COPY host /host

RUN . $HOME/.cargo/env && cargo install tinychain --path=/host

RUN mkdir /data

RUN ln -s $HOME/.cargo/bin/tinychain tinychain
