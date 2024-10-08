#!/bin/bash

# load UBUNTU_CODENAME by sourcing the os-release config file
. /etc/os-release

if [ -z $UBUNTU_CODENAME ]
then
    echo "install.sh only supports Ubuntu Linux"
    echo "for other operating systems, see the manual install instructions in INSTALL.md"
    exit
fi

# parse arguments
while [ $# -gt 0 ]; do

   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
   fi

  shift
done

# remind the user to install CUDA, if supported
if command -v lspci &> /dev/null
then
    gpu_available=`lspci | grep -i nvidia | wc -l`
    if (( $gpu_available > 0 ))
    then
        if ! dpkg-query -l cuda > /dev/null 2>&1
        then
            echo "your system may support GPU acceleration with CUDA, but CUDA does not appear to be installed"
            echo "CUDA install instructions: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation"
        fi
    fi
fi

# check for missing dependencies for this script
while ! command -v curl &> /dev/null
do
    echo "installing cURL"
    apt-get -y install curl
done

# make sure cargo is installed
while ! command -v cargo &> /dev/null
do
    echo "installing cargo (more info: https://doc.rust-lang.org/cargo/)"
    curl https://sh.rustup.rs -sSf | sh -s -- -y
    source $HOME/.cargo/env
    echo "installed cargo"
done

# make sure the C linker is installed
while ! dpkg-query -l build-essential > /dev/null 2>&1
do
    echo "installing build tools"
    apt-get install -y build-essential
done

while ! command -v tinychain &> /dev/null
do
    # install TinyChain
    echo "installing TinyChain"
    cargo install tinychain
done

echo 'TinyChain installed successfully--remember to run source $HOME/.cargo/env before running the tinychain command'
