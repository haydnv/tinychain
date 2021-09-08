This crate is used internally by TinyChain. It provides generic Id, Map, and Tuple types.

Example:
```rust
use safecast::TryCastFrom;
use tcgeneric::{Id, Map, Tuple};

let tuple = Tuple::<(Id, String)>::from_iter(vec![]);
assert_eq!(Map::opt_cast_from(tuple).unwrap(), Map::default());
```

For more information on TinyChain, see: [http://github.com/haydnv/tinychain](http://github.com/haydnv/tinychain)
