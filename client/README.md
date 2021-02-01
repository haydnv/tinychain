# Tinychain Python Client

For details about the Tinychain application runtime, see
[http://github.com/haydnv/tinychain](http://github.com/haydnv/tinychain).

## Getting started
First, make sure you have Tinychain installed. The easiest way to do this is to use the Docker
image available here:
[https://hub.docker.com/r/haydnv/tinychain](https://hub.docker.com/r/haydnv/tinychain)

```
docker pull haydnv/tinychain:demo2

# include the "--gpus all" argument if you have nvidia-docker installed
docker run -it --rm haydnv/tinychain:demo2 bash
source ~/.profile
cd ~/tinychain
```

If you're not using the Docker image, you will need to set these environment variables:

 * `AF_PATH`: path to the Arrayfire library (if running a debug version of Tinychain)
 * `LD_LIBRARY_PATH`: set this to include AF_PATH (if running a debug version of Tinychain)
 * `TINYCHAIN_PATH`: path to the Tinychain executable

Then, try this hello world program:

```
$ ipython3
...
# load the Python client
import tinychain as tc

# if you're not using the Docker image, you'll need to update this accordingly
PATH = "/tinychain/target/debug/tinychain"

# start a new Tinychain host on port 8888 using directory /tmp/helloworld
host = tc.LocalHost(PATH, "helloworld", 8888)
host.get("/sbin/value/string/ustring", "Hello, World!")
```

Look through the [examples](https://github.com/haydnv/tctest/tree/master/examples) test directory
for more Python client usage examples.
