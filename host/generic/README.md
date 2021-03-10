This crate is used internally by Tinychain. It provides generic Id, Map, and Tuple types.

Example:
```rust
use safecast::TryCastFrom;
use tcgeneric::{Id, Map, Tuple};

let tuple = Tuple::<(Id, String)>::from_iter(vec![]);
assert_eq!(Map::opt_cast_from(tuple).unwrap(), Map::default());
```

For more information on Tinychain, see: [http://github.com/haydnv/tinychain](http://github.com/haydnv/tinychain)
