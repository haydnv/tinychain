## How to Contribute

We're excited to accept your contributions to TinyChain! If you're not sure where to get started,
consider [starting a discussion](https://github.com/haydnv/tinychain/discussions)
or [creating an issue](https://github.com/haydnv/tinychain/issues).

If you already have some changes you'd like to submit, just send a pull request.
You can find instructions for creating a pull request in
[GitHub Help](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests).

## New Developer Quickstart Guide

Developing TinyChain itself is a little different than developing services that run on TinyChain.
For example, you won't be able to use the TinyChain client from `pip` because you'll be making changes to the client itself!
The best way to get started is to clone the git repository:

```bash
git clone https://github.com/haydnv/tinychain
```

If you prefer, you can [fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/about-forks) the repo instead. That way, you'll be able to save your commits to GitHub before you contribute them upstream.

To run the client unit tests, make sure Python knows where to find your local copy of the TinyChain client:

```bash
# for example, if you cloned into /home/usernames/tinychain
# then the path below should be /home/username/tinychain/client
export PYTHONPATH=$PYTHONPATH:/<path to your clone of the repo>/client
```

Then, try running the client tests to make sure you've got everything configured correctly:

```
python3 tests/client
```

There are some more tests in the `tests/host` directory--don't worry about those yet. They require starting multiple hosts programmatically, which is covered in the next section on "Rust Development".

By default the client tests will run on `demo.tinychain.net` but if you do much development on TinyChain you'll probably want to run your own host. You can do that using Docker:

```bash
# you can change the --git URL to your own if you need
docker build --build-arg "CRATE=--git=https://github.com/haydnv/tinychain.git" .

# the docker build command will output an image ID when it finishes
# use that to run the image you just build

# --address=0.0.0.0 tells the TinyChain process inside the Docker container to listen on any address
# you can also add other startup parameters like --http_port, --cache_size, --help, etc
docker run -d -p 127.0.0.1:8702:8702/tcp <image ID> /tinychain --data_dir=/data --address=0.0.0.0
```

Check that your Dockerized TinyChain host is up and running by visiting [http://localhost:8702/state/scalar/value/string?key=%22Hello,%20World!%22](http://localhost:8702/state/scalar/value/string?key=%22Hello,%20World!%22) in your browser.

Now that you have your own host running, you can run your client tests against it:

```bash
# tell the test suite what host to contact
export TC_HOST=http://127.0.0.1:8702

# run the tests
python3 tests/client
```

## Rust Development

If you need to make changes to the TinyChain host software itself, you'll need to run it locally. Follow the steps in the "Manual Install" section of `INSTALL.md` to set up Rust and ArrayFire, then check that you've gotten everything set up correctly by running the host tests:

```bash
python3 tests/host
```

If you have any problems, first check that you have all your environment variables set correctly. If `PYTHONPATH`, `AF_PATH`, and `LD_LIBRARY_PATH` are all set correctly and your installation still doesn't behave as expected, [create an issue](https://github.com/haydnv/tinychain/issues) to get help.

## Licensing

By contributing code to the TinyChain project, you represent that you own the copyright on your contributions, or that you have followed the licensing requirements of the copyright holder, and that TinyChain may use your code without any further restrictions than those specified in the Apache 2.0 open-source license. A copy of the license can be found in the `LICENSE` file in the root directory of the project.

## Code of Conduct

This project follows the [IndieWeb code of conduct](https://indieweb.org/code-of-conduct).
