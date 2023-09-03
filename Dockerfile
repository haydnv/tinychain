FROM ubuntu:20.04
LABEL Name=tinychain Version=0.0.2
ARG CRATE=""
ARG TZ=America/New_York

ENV TZ=${TZ}

ENV CRATE=${CRATE}

RUN echo $TZ $CRATE

RUN apt-get -y update && apt-get install -y sudo curl

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

RUN apt-get install -y gnupg2 ca-certificates apt-utils software-properties-common

RUN apt-get install -y build-essential

RUN . $HOME/.cargo/env && cargo install tinychain $CRATE

RUN ln -s $HOME/.cargo/bin/tinychain tinychain --data_dir=/tmp/data
