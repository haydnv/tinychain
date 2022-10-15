//! A [`TxnLock`] to support transaction-specific versioning

mod map;
mod scalar;

pub use map::{
    Keys, TxnMapLock, TxnMapLockCommitGuard, TxnMapLockReadGuard, TxnMapLockReadGuardExclusive,
    TxnMapLockWriteGuard, TxnMapRead, TxnMapWrite,
};

pub use scalar::{
    TxnLock, TxnLockCommitGuard, TxnLockReadGuard, TxnLockReadGuardExclusive, TxnLockWriteGuard,
};
