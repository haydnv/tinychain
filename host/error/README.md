This crate is used internally by Tinychain. It provides a generic error type `TCError` which can be mapped to common HTTP error codes and supports serialization and deserialization with `destream`.

Example:
```rust
use tc_error::*;

fn expect_true(value: bool) -> TCResult<()> {
    if value {
        Ok(())
    } else {
        Err(TCError::bad_request("expected true but found", value))
    }
}

assert_eq!(expect_true(true), Ok(()));
```

For more information on Tinychain, see: [http://github.com/haydnv/tinychain](http://github.com/haydnv/tinychain)
