This is the automated test suite for TinyChain. It includes utilites to start and interact with one or more TinyChain hosts via Docker or a binary distribution.

For more information about TinyChain, see [https://www.tinychain.net/](https://www.tinychain.net/).

The test suite must be run from the main project directory in order to make the `host` directory available in the Docker build environment.

Example usage:
```bash
# python3 and pip3 may be python and pip on newer systems
pip3 install -e tests
python3 -m tests.tctest.client
python3 -m tests.tctest.apps -k Library
python3 -m tests.tctest # run all tests
```
