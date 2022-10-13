//! A [`TxnLock`] to support transaction-specific versioning

mod map;
mod scalar;

pub use map::{
    TxnMapLock, TxnMapLockReadGuard, TxnMapLockReadGuardExclusive, TxnMapLockWriteGuard,
    TxnMapRead, TxnMapWrite,
};

pub use scalar::{TxnLock, TxnLockReadGuard, TxnLockReadGuardExclusive, TxnLockWriteGuard};
