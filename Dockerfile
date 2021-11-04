FROM rust
LABEL Name=tinychain Version=0.0.1
RUN apt-get -y update && apt-get install -y sudo

# Timezone Setting
ARG TZ=America/New_York

# Build Argument TZ. Default Value: New New_York
# Pass the TZ variable as --build-arg to docker build command to set your preference for the time zone
ENV TZ=${TZ}
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt install -y wget

WORKDIR /tmp

RUN wget https://raw.githubusercontent.com/haydnv/tinychain/master/install.sh | sh

ENTRYPOINT ["tinychain"]
