import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tinychain-haydnv",
    version="0.1.0",
    author="Haydn Vestal",
    description="A Python client for Tinychain",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/haydnv/tinychain.py",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache 2",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
