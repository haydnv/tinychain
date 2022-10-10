//! A [`TxnLock`] to support transaction-specific versioning

use std::collections::HashMap;
use std::ops::Deref;
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::TxnId;

mod scalar;

pub use scalar::{TxnLock, TxnLockReadGuard, TxnLockReadGuardExclusive, TxnLockWriteGuard};

#[derive(Copy, Clone)]
struct Wake;

struct Versions<T> {
    canon: T,
    versions: HashMap<TxnId, Arc<RwLock<T>>>,
}

impl<T: Clone + PartialEq<T>> Versions<T> {
    fn commit(&mut self, txn_id: &TxnId) -> bool {
        if let Some(version) = self.versions.get(txn_id) {
            let version = version.try_read().expect("transaction version");
            if version.deref() == &self.canon {
                // no-op
                false
            } else {
                self.canon = version.clone();
                true
            }
        } else {
            // it's still valid to read the version at this transaction, so keep a copy around
            let canon = self.canon.clone();
            self.versions.insert(*txn_id, Arc::new(RwLock::new(canon)));
            false
        }
    }

    fn finalize(&mut self, txn_id: &TxnId) {
        self.versions.remove(txn_id);
    }

    fn get(self: &mut Versions<T>, txn_id: TxnId) -> Arc<RwLock<T>> {
        if let Some(version) = self.versions.get(&txn_id) {
            version.clone()
        } else {
            let version = self.canon.clone();
            let version = Arc::new(RwLock::new(version));
            self.versions.insert(txn_id, version.clone());
            version
        }
    }
}
