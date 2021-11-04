FROM ubuntu:20.04
LABEL Name=tinychain Version=0.0.2
RUN apt-get -y update && apt-get install -y sudo curl

# Timezone Setting
ARG TZ=America/New_York

# Build Argument TZ. Default Value: New New_York
# Pass the TZ variable as --build-arg to docker build command to set your preference for the time zone
ENV TZ=${TZ}
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

RUN apt-get install -y gnupg2 ca-certificates apt-utils software-properties-common

RUN apt-get install -y build-essential

RUN apt-key adv --fetch-key https://repo.arrayfire.com/GPG-PUB-KEY-ARRAYFIRE-2020.PUB

RUN echo "deb [arch=amd64] https://repo.arrayfire.com/ubuntu focal main" | tee /etc/apt/sources.list.d/arrayfire.list

RUN apt-get update && apt-get install -y arrayfire

RUN . $HOME/.cargo/env && cargo install tinychain --features=tensor

ENTRYPOINT ["tinychain"]
