//! A [`TxnMapLock`] to support transaction-specific versioning of a collection of states.

use std::borrow::Borrow;
use std::collections::hash_map::{Entry, HashMap};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use log::debug;

use tc_error::*;
use tcgeneric::{Id, Map};

use crate::lock::{
    TxnLock, TxnLockCommitGuard, TxnLockReadGuard, TxnLockReadGuardExclusive, TxnLockWriteGuard,
};
use crate::{Transact, TxnId};

trait Guard<T> {
    fn id(&self) -> &TxnId;

    fn keys(&self) -> &BTreeSet<Id>;

    fn values(&self) -> &Arc<Mutex<BTreeMap<Id, Value<T>>>>;
}

pub struct Iter<'a, T> {
    txn_id: TxnId,
    keys: std::collections::btree_set::Iter<'a, Id>,
    values: Arc<Mutex<BTreeMap<Id, Value<T>>>>,
}

impl<'a, T: Clone> Iterator for Iter<'a, T> {
    type Item = (&'a Id, T);

    fn next(&mut self) -> Option<Self::Item> {
        let key = self.keys.next()?;
        let mut values = self.values.lock().expect("TxnMapLock state");
        let value = values.get_mut(key)?;
        Some((key, value.read(self.txn_id)))
    }
}

pub trait TxnMapRead<T> {
    fn contains_key<K: Borrow<Id>>(&self, key: K) -> bool;

    fn get<K: Borrow<Id>>(&self, key: K) -> Option<T>;

    fn iter(&self) -> Iter<T>;

    fn is_empty(&self) -> bool;
}

impl<G, T: Clone> TxnMapRead<T> for G
where
    G: Guard<T>,
{
    fn contains_key<K: Borrow<Id>>(&self, key: K) -> bool {
        self.keys().contains(key.borrow())
    }

    fn get<K: Borrow<Id>>(&self, key: K) -> Option<T> {
        let mut values = self.values().lock().expect("TxnMapLock state");
        let version = values.get_mut(key.borrow())?;
        Some(version.read(*self.id()))
    }

    fn iter(&self) -> Iter<T> {
        Iter {
            txn_id: *self.id(),
            keys: self.keys().iter(),
            values: self.values().clone(),
        }
    }

    fn is_empty(&self) -> bool {
        self.keys().is_empty()
    }
}

pub trait TxnMapWrite<T>: TxnMapRead<T> {
    fn drain(&mut self) -> Drain<T>;

    fn insert(&mut self, key: Id, value: T) -> bool;

    fn remove<K: Borrow<Id>>(&mut self, key: K) -> bool;
}

#[derive(Clone)]
pub struct TxnMapLockReadGuard<T> {
    lock: TxnMapLock<T>,
    guard: TxnLockReadGuard<BTreeSet<Id>>,
}

impl<T: Clone> Guard<T> for TxnMapLockReadGuard<T> {
    fn id(&self) -> &TxnId {
        self.guard.id()
    }

    fn keys(&self) -> &BTreeSet<Id> {
        &*self.guard
    }

    fn values(&self) -> &Arc<Mutex<BTreeMap<Id, Value<T>>>> {
        &self.lock.values
    }
}

pub struct TxnMapLockReadGuardExclusive<T> {
    lock: TxnMapLock<T>,
    guard: TxnLockReadGuardExclusive<BTreeSet<Id>>,
}

impl<T> TxnMapLockReadGuardExclusive<T> {
    pub fn upgrade(self) -> TxnMapLockWriteGuard<T> {
        TxnMapLockWriteGuard {
            lock: self.lock,
            guard: self.guard.upgrade(),
        }
    }
}

impl<T> Guard<T> for TxnMapLockReadGuardExclusive<T> {
    fn id(&self) -> &TxnId {
        self.guard.id()
    }

    fn keys(&self) -> &BTreeSet<Id> {
        &*self.guard
    }

    fn values(&self) -> &Arc<Mutex<BTreeMap<Id, Value<T>>>> {
        &self.lock.values
    }
}

pub struct TxnMapLockWriteGuard<T> {
    lock: TxnMapLock<T>,
    guard: TxnLockWriteGuard<BTreeSet<Id>>,
}

impl<T> TxnMapLockWriteGuard<T> {
    pub fn downgrade(self) -> TxnMapLockReadGuardExclusive<T> {
        TxnMapLockReadGuardExclusive {
            lock: self.lock,
            guard: self.guard.downgrade(),
        }
    }
}

impl<T: Clone> Guard<T> for TxnMapLockWriteGuard<T> {
    fn id(&self) -> &TxnId {
        self.guard.id()
    }

    fn keys(&self) -> &BTreeSet<Id> {
        &*self.guard
    }

    fn values(&self) -> &Arc<Mutex<BTreeMap<Id, Value<T>>>> {
        &self.lock.values
    }
}

pub struct Drain<'a, T> {
    txn_id: TxnId,
    drain_from: Vec<Id>,
    keys: &'a mut BTreeSet<Id>,
    values: Arc<Mutex<BTreeMap<Id, Value<T>>>>,
}

impl<'a, T: Clone> Iterator for Drain<'a, T> {
    type Item = (Id, T);

    fn next(&mut self) -> Option<Self::Item> {
        let key = self.drain_from.pop()?;
        self.keys.remove(&key);

        let mut values = self.values.lock().expect("TxnMapLock state");
        let value = values.get_mut(&key)?;

        Some((key, value.read(self.txn_id)))
    }
}

impl<T: Clone> TxnMapWrite<T> for TxnMapLockWriteGuard<T> {
    fn drain(&mut self) -> Drain<T> {
        Drain {
            txn_id: *self.guard.id(),
            drain_from: self.guard.iter().rev().cloned().collect::<Vec<Id>>(),
            keys: &mut *self.guard,
            values: self.lock.values.clone(),
        }
    }

    fn insert(&mut self, key: Id, value: T) -> bool {
        let mut values = self.lock.values.lock().expect("TxnMapLock state");

        if self.guard.insert(key.clone()) {
            values.insert(key, Value::version(*self.guard.id(), value));
            false
        } else {
            let version = values.get_mut(&key).expect("value version");
            version.write(*self.guard.id(), value);
            true
        }
    }

    fn remove<K: Borrow<Id>>(&mut self, key: K) -> bool {
        self.guard.remove(key.borrow())
    }
}

pub struct TxnMapLockCommitGuard<T> {
    keys: TxnLockCommitGuard<BTreeSet<Id>>,
    values: Arc<Mutex<BTreeMap<Id, Value<T>>>>,
}

impl<T> Guard<T> for TxnMapLockCommitGuard<T> {
    fn id(&self) -> &TxnId {
        self.keys.id()
    }

    fn keys(&self) -> &BTreeSet<Id> {
        &self.keys
    }

    fn values(&self) -> &Arc<Mutex<BTreeMap<Id, Value<T>>>> {
        &self.values
    }
}

struct Value<T> {
    canon: Option<T>,
    versions: HashMap<TxnId, T>,
}

impl<T> Value<T> {
    fn canon(canon: T) -> Self {
        Self {
            canon: Some(canon),
            versions: HashMap::new(),
        }
    }

    fn version(txn_id: TxnId, version: T) -> Self {
        let mut versions = HashMap::new();
        versions.insert(txn_id, version);

        Self {
            canon: None,
            versions,
        }
    }
}

impl<T: Clone> Value<T> {
    fn commit(&mut self, txn_id: &TxnId) {
        if let Some(value) = self.versions.get(txn_id) {
            self.canon = Some(value.clone());
        }
    }

    fn finalize(&mut self, txn_id: &TxnId) {
        self.versions.remove(txn_id);
    }

    fn read(&mut self, txn_id: TxnId) -> T {
        if let Some(value) = self.versions.get(&txn_id) {
            value.clone()
        } else {
            let value = self.canon.clone().expect("canonical value");
            self.versions.insert(txn_id, value.clone());
            value
        }
    }

    fn write(&mut self, txn_id: TxnId, value: T) {
        match self.versions.entry(txn_id) {
            Entry::Occupied(mut entry) => {
                *entry.get_mut() = value;
            }
            Entry::Vacant(entry) => {
                entry.insert(value);
            }
        }
    }
}

#[derive(Clone)]
pub struct TxnMapLock<T> {
    name: Arc<String>,
    keys: TxnLock<BTreeSet<Id>>,
    values: Arc<Mutex<BTreeMap<Id, Value<T>>>>,
}

impl<T> TxnMapLock<T> {
    pub fn new<I: fmt::Display>(name: I) -> Self {
        Self {
            name: Arc::new(name.to_string()),
            keys: TxnLock::new(format!("{} keys", name), BTreeSet::new()),
            values: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }

    pub fn with_contents<I: fmt::Display>(name: I, contents: Map<T>) -> Self {
        let mut keys = BTreeSet::new();
        let mut values = BTreeMap::new();
        for (k, v) in contents.into_iter() {
            keys.insert(k.clone());
            values.insert(k, Value::canon(v));
        }

        Self {
            name: Arc::new(name.to_string()),
            keys: TxnLock::new(format!("{} keys", name), keys),
            values: Arc::new(Mutex::new(values)),
        }
    }
}

impl<T: Clone> TxnMapLock<T> {
    /// Lock this map for reading at the given `txn_id`.
    pub async fn read(&self, txn_id: TxnId) -> TCResult<TxnMapLockReadGuard<T>> {
        debug!("lock map {} to read at {}...", self.name, txn_id);

        let guard = self.keys.read(txn_id).await?;
        let guard = TxnMapLockReadGuard {
            lock: self.clone(),
            guard,
        };

        debug!("locked map {} for reading at {}", self.name, txn_id);
        Ok(guard)
    }

    /// Lock this map for reading at the given `txn_id`, if possible.
    pub fn try_read(&self, txn_id: TxnId) -> TCResult<TxnMapLockReadGuard<T>> {
        debug!("lock map {} to read at {}...", self.name, txn_id);

        let guard = self.keys.try_read(txn_id)?;
        let guard = TxnMapLockReadGuard {
            lock: self.clone(),
            guard,
        };

        debug!("locked map {} for reading at {}", self.name, txn_id);
        Ok(guard)
    }

    /// Lock this map exclusively for reading at the given `txn_id`.
    pub async fn read_exclusive(&self, txn_id: TxnId) -> TCResult<TxnMapLockReadGuardExclusive<T>> {
        debug!(
            "lock map {} exclusively to read at {}...",
            self.name, txn_id
        );

        let guard = self.keys.read_exclusive(txn_id).await?;
        let guard = TxnMapLockReadGuardExclusive {
            lock: self.clone(),
            guard,
        };

        debug!(
            "locked map {} exclusively for reading at {}",
            self.name, txn_id
        );

        Ok(guard)
    }

    /// Lock this map for writing at the given `txn_id`.
    pub async fn write(&self, txn_id: TxnId) -> TCResult<TxnMapLockWriteGuard<T>> {
        debug!("lock map {} for writing at {}...", self.name, txn_id);

        let guard = self.keys.write(txn_id).await?;
        let guard = TxnMapLockWriteGuard {
            lock: self.clone(),
            guard,
        };

        debug!("locked {} for writing at {}", self.name, txn_id);
        Ok(guard)
    }
}

#[async_trait]
impl<T: Clone + Send + Sync> Transact for TxnMapLock<T> {
    type Commit = TxnMapLockCommitGuard<T>;

    async fn commit(&self, txn_id: &TxnId) -> Self::Commit {
        debug!("commit map {} at {}", self.name, txn_id);

        let keys = self.keys.commit(txn_id).await;
        let mut values = self.values.lock().expect("transactional map values");

        for key in &*keys {
            let value = values.get_mut(key).expect("transactional map value");
            value.commit(txn_id);
        }

        TxnMapLockCommitGuard {
            keys,
            values: self.values.clone(),
        }
    }

    async fn finalize(&self, txn_id: &TxnId) {
        debug!("finalize map {} at {}", self.name, txn_id);

        let keys = self
            .keys
            .read_exclusive(*txn_id)
            .await
            .expect("transactional map keys");

        {
            let mut values = self.values.lock().expect("transactional map values");

            let mut to_delete = Vec::with_capacity(values.len());
            for (key, value) in values.iter_mut() {
                if keys.contains(key) {
                    value.finalize(txn_id);
                } else {
                    if let Some(last_commit) = value.versions.keys().last() {
                        if last_commit <= txn_id {
                            to_delete.push(key.clone());
                        }
                    } else {
                        to_delete.push(key.clone())
                    }
                }
            }

            for key in to_delete.into_iter() {
                values.remove(&key);
            }
        }

        self.keys.finalize(txn_id).await;
    }
}
