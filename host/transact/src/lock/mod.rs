//! A [`TxnLock`] to support transaction-specific versioning

mod map;
mod scalar;

pub use map::{
    TxnMapLock, TxnMapLockCommitGuard, TxnMapLockReadGuard, TxnMapLockReadGuardExclusive,
    TxnMapLockWriteGuard, TxnMapRead, TxnMapWrite,
};

pub use scalar::{
    TxnLock, TxnLockCommitGuard, TxnLockReadGuard, TxnLockReadGuardExclusive, TxnLockWriteGuard,
};
