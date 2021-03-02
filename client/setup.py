import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tinychain",
    version="0.1.5",
    author="Haydn Vestal",
    description="A Python client for Tinychain",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/haydnv/tinychain",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Software Development",
        "Topic :: Software Development :: Code Generators",
    ],
    python_requires='>=3.7',
)
