FROM ubuntu:20.04
LABEL Name=tinychain Version=0.0.2
ARG CRATE=""
ARG TZ=America/New_York

ENV TZ=${TZ}

ENV CRATE=${CRATE}

RUN echo $TZ $CRATE

RUN apt-get -y update && apt-get install -y sudo curl

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get install -y gnupg2 ca-certificates apt-utils software-properties-common

RUN apt-get install -y build-essential

WORKDIR /tmp

RUN curl -sSL https://arrayfire.s3.amazonaws.com/3.8.0/ArrayFire-v3.8.0_Linux_x86_64.sh --output ArrayFire-v3.8.0_Linux_x86_64.sh && \
    chmod +x ArrayFire-v3.8.0_Linux_x86_64.sh && \
    bash ArrayFire-v3.8.0_Linux_x86_64.sh --include-subdir --prefix=/opt --skip-license && \
    rm -rf /tmp/ArrayFire-*

RUN sh -c "echo '/opt/arrayfire/lib64' > /etc/ld.so.conf.d/arrayfire.conf" ldconfig

ENV AF_PATH=/opt/arrayfire
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$AF_PATH/lib64
ENV PKG_CONFIG_PATH=/root

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

RUN . $HOME/.cargo/env && cargo install tinychain --features=tensor $CRATE

WORKDIR /

RUN ln -s $HOME/.cargo/bin/tinychain tinychain
