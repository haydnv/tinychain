This crate is used internally by TinyChain to provide the Tensor data structure. You can enable it using the `tensor` feature (e.g. `cargo build --release --features=tensor`). It requires linking to [ArrayFire](http://arrayfire.com) version 3.8 in order to compile. You can download and install ArrayFire by following the instructions at [http://arrayfire.org/docs/installing.htm](http://arrayfire.org/docs/installing.htm). You'll also have to add a package config file to your `$PKG_CONFIG_PATH` like so:

```
prefix=/usr
exec_prefix=${prefix}
includedir=${prefix}/include
libdir=${exec_prefix}/lib64

Name: arrayfire
Description: the ArrayFire library
Version: 3.8
Libs: -L${libdir}
```

For more information on TinyChain, see [http://github.com/haydnv/tinychain](http://github.com/haydnv/tinychain)
