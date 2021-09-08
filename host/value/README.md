This crate is used internally by TinyChain. It provides a generic Value type which supports (de)serialization with [serde](https://docs.rs/serde/) and [destream](https://docs.rs/destream/) as well as equality and collation.

Example:
```rust
use safecast::CastFrom;
use tcgeneric::Tuple;

let row = Value::cast_from(("name", 12345));
assert_eq!(row, Value::Tuple(Tuple::from(vec![Value::String("name"), Value::Number(12345.into())])));
```

For more information on TinyChain, see [http://github.com/haydnv/tinychain](http://github.com/haydnv/tinychain)
