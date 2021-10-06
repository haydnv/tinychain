FROM rust
LABEL Name=tinychain Version=0.0.1
RUN apt-get -y update && apt-get install -y sudo

# Timezone Setting
ARG TZ=America/New_York

# Build Argument TZ. Default Value: New New_York
# Pass the TZ variable as --build-arg to docker build command to set your preference for the time zone
ENV TZ=${TZ}
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get install -y python3 python3-venv python3-pip make build-essential libssl-dev zlib1g-dev libbz2-dev libfreeimage3 libfontconfig1 libglu1-mesa \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev pkg-config \
    xz-utils tk-dev libffi-dev liblzma-dev git && \
    curl https://pyenv.run | bash 

WORKDIR /tmp

RUN curl -sSL https://arrayfire.s3.amazonaws.com/3.8.0/ArrayFire-v3.8.0_Linux_x86_64.sh --output ArrayFire-v3.8.0_Linux_x86_64.sh && \ 
    chmod +x ArrayFire-v3.8.0_Linux_x86_64.sh && \
    bash ArrayFire-v3.8.0_Linux_x86_64.sh --include-subdir --prefix=/opt --skip-license && \
    rm -rf /tmp/ArrayFire-*

RUN sh -c "echo '/opt/arrayfire/lib64' > /etc/ld.so.conf.d/arrayfire.conf" ldconfig

ENV AF_PATH=/opt/arrayfire
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$AF_PATH/lib64
ENV PKG_CONFIG_PATH=/root

# Install Tinychain with tensor feature
RUN cargo install tinychain --features=tensor

ENV HOME=/root
ENV PYENV_ROOT=$HOME/.pyenv
