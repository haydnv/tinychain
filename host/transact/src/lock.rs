use super::TxnId;

pub use txn_lock::TxnLockWriteGuard;

pub type TxnLock<T> = txn_lock::TxnLock<TxnId, T>;
pub type TxnLockReadGuard<T> = txn_lock::TxnLockReadGuard<TxnId, T>;
pub type TxnLockReadGuardExclusive<T> = txn_lock::TxnLockReadGuardExclusive<TxnId, T>;
