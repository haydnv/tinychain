This crate is used internally by Tinychain. It provides traits and data structures to support transactional mutations of in-memory and persistent datatypes.

Example:
```rust
use tc_transact::{TxnId, TxnLock};

let version = TxnLock::new("version", 0);

let txn_one = TxnId::new(1);
let txn_two = TxnId::new(2);
let txn_three = TxnId::new(3);

assert_eq!(version.read(txn_one).await.unwrap(), 0);

*(version.write(txn_two).await.unwrap()) = 2;
version.commit(txn_two).await;

assert_eq!(version.read(txn_three).await.unwrap(), 2);

```

For more information on Tinychain, see [http://github.com/haydnv/tinychain](http://github.com/haydnv/tinychain)
