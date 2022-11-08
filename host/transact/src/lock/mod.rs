//! A [`TxnLock`] to support transaction-specific versioning

pub mod map;
pub mod scalar;

pub use map::{
    Iter, Keys, TxnMapLock, TxnMapLockCommitGuard, TxnMapLockReadGuard,
    TxnMapLockReadGuardExclusive, TxnMapLockWriteGuard, TxnMapRead, TxnMapWrite,
};

pub use scalar::{
    TxnLock, TxnLockCommitGuard, TxnLockReadGuard, TxnLockReadGuardExclusive, TxnLockWriteGuard,
};
