//! A [`TxnMapLock`] to support transaction-specific versioning of a collection of states.

use std::borrow::Borrow;
use std::collections::hash_map::{Entry, HashMap};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use log::debug;

use tc_error::*;

use crate::lock::{
    TxnLock, TxnLockCommitGuard, TxnLockReadGuard, TxnLockReadGuardExclusive, TxnLockWriteGuard,
};
use crate::{Transact, TxnId};

trait Guard<K, V> {
    fn borrow(&self) -> &BTreeSet<K>;

    fn id(&self) -> &TxnId;

    fn values(&self) -> &Arc<Mutex<BTreeMap<K, Value<V>>>>;
}

pub struct Iter<'a, K, V> {
    txn_id: TxnId,
    keys: std::collections::btree_set::Iter<'a, K>,
    values: Arc<Mutex<BTreeMap<K, Value<V>>>>,
}

impl<'a, K, V> Iterator for Iter<'a, K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    type Item = (&'a K, V);

    fn next(&mut self) -> Option<Self::Item> {
        let key = self.keys.next()?;
        let mut values = self.values.lock().expect("TxnMapLock state");
        let value = values.get_mut(key)?;
        Some((key, value.read(self.txn_id)))
    }
}

pub struct Keys<'a, K> {
    iter: std::collections::btree_set::Iter<'a, K>,
}

impl<'a, K> Iterator for Keys<'a, K> {
    type Item = &'a K;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.iter.count()
    }

    fn last(self) -> Option<Self::Item>
    where
        Self: Sized,
    {
        self.iter.last()
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.iter.nth(n)
    }

    fn max(self) -> Option<Self::Item>
    where
        Self: Sized,
        Self::Item: Ord,
    {
        self.iter.max()
    }

    fn min(self) -> Option<Self::Item>
    where
        Self: Sized,
        Self::Item: Ord,
    {
        self.iter.min()
    }
}

pub trait TxnMapRead<K, V> {
    fn contains_key<Q: Borrow<K>>(&self, key: Q) -> bool;

    fn get<Q: Borrow<K>>(&self, key: Q) -> Option<V>;

    fn iter(&self) -> Iter<K, V>;

    fn is_empty(&self) -> bool;

    fn keys(&self) -> Keys<K>;

    fn len(&self) -> usize;
}

impl<G, K, V> TxnMapRead<K, V> for G
where
    G: Guard<K, V>,
    K: Ord + Clone,
    V: Clone,
{
    fn contains_key<Q: Borrow<K>>(&self, key: Q) -> bool {
        self.borrow().contains(key.borrow())
    }

    // TODO: this should return a borrowed value
    fn get<Q: Borrow<K>>(&self, key: Q) -> Option<V> {
        let mut values = self.values().lock().expect("TxnMapLock state");
        let version = values.get_mut(key.borrow())?;
        Some(version.read(*self.id()))
    }

    fn iter(&self) -> Iter<K, V> {
        Iter {
            txn_id: *self.id(),
            keys: self.borrow().iter(),
            values: self.values().clone(),
        }
    }

    fn is_empty(&self) -> bool {
        self.borrow().is_empty()
    }

    fn keys(&self) -> Keys<K> {
        Keys {
            iter: self.borrow().iter(),
        }
    }

    fn len(&self) -> usize {
        self.borrow().len()
    }
}

pub trait TxnMapWrite<K, V>: TxnMapRead<K, V> {
    fn drain(&mut self) -> Drain<K, V>;

    fn insert(&mut self, key: K, value: V) -> bool;

    fn remove<Q: Borrow<K>>(&mut self, key: Q) -> bool;
}

#[derive(Clone)]
pub struct TxnMapLockReadGuard<K: PartialEq + Clone, V> {
    lock: TxnMapLock<K, V>,
    guard: TxnLockReadGuard<BTreeSet<K>>,
}

impl<K, V> Guard<K, V> for TxnMapLockReadGuard<K, V>
where
    K: PartialEq + Clone,
    V: Clone,
{
    fn borrow(&self) -> &BTreeSet<K> {
        &*self.guard
    }

    fn id(&self) -> &TxnId {
        self.guard.id()
    }

    fn values(&self) -> &Arc<Mutex<BTreeMap<K, Value<V>>>> {
        &self.lock.values
    }
}

pub struct TxnMapLockReadGuardExclusive<K: PartialEq + Clone, V> {
    lock: TxnMapLock<K, V>,
    guard: TxnLockReadGuardExclusive<BTreeSet<K>>,
}

impl<K, V> TxnMapLockReadGuardExclusive<K, V>
where
    K: PartialEq + Clone,
{
    pub fn upgrade(self) -> TxnMapLockWriteGuard<K, V> {
        TxnMapLockWriteGuard {
            lock: self.lock,
            guard: self.guard.upgrade(),
        }
    }
}

impl<K, V> Guard<K, V> for TxnMapLockReadGuardExclusive<K, V>
where
    K: PartialEq + Clone,
{
    fn borrow(&self) -> &BTreeSet<K> {
        &*self.guard
    }

    fn id(&self) -> &TxnId {
        self.guard.id()
    }

    fn values(&self) -> &Arc<Mutex<BTreeMap<K, Value<V>>>> {
        &self.lock.values
    }
}

pub struct TxnMapLockWriteGuard<K: PartialEq + Clone, V> {
    lock: TxnMapLock<K, V>,
    guard: TxnLockWriteGuard<BTreeSet<K>>,
}

impl<K, V> TxnMapLockWriteGuard<K, V>
where
    K: PartialEq + Clone,
{
    pub fn downgrade(self) -> TxnMapLockReadGuardExclusive<K, V> {
        TxnMapLockReadGuardExclusive {
            lock: self.lock,
            guard: self.guard.downgrade(),
        }
    }
}

impl<K, V> Guard<K, V> for TxnMapLockWriteGuard<K, V>
where
    K: PartialEq + Clone,
    V: Clone,
{
    fn borrow(&self) -> &BTreeSet<K> {
        &*self.guard
    }

    fn id(&self) -> &TxnId {
        self.guard.id()
    }

    fn values(&self) -> &Arc<Mutex<BTreeMap<K, Value<V>>>> {
        &self.lock.values
    }
}

pub struct Drain<'a, K, V> {
    txn_id: TxnId,
    drain_from: Vec<K>,
    keys: &'a mut BTreeSet<K>,
    values: Arc<Mutex<BTreeMap<K, Value<V>>>>,
}

impl<'a, K, V> Iterator for Drain<'a, K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        let key = self.drain_from.pop()?;
        self.keys.remove(&key);

        let mut values = self.values.lock().expect("TxnMapLock state");
        let value = values.get_mut(&key)?;

        Some((key, value.read(self.txn_id)))
    }
}

impl<K, V> TxnMapWrite<K, V> for TxnMapLockWriteGuard<K, V>
where
    K: Ord + PartialEq + Clone,
    V: Clone,
{
    fn drain(&mut self) -> Drain<K, V> {
        Drain {
            txn_id: *self.guard.id(),
            drain_from: self.guard.iter().rev().cloned().collect::<Vec<K>>(),
            keys: &mut *self.guard,
            values: self.lock.values.clone(),
        }
    }

    fn insert(&mut self, key: K, value: V) -> bool {
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

    fn remove<Q: Borrow<K>>(&mut self, key: Q) -> bool {
        self.guard.remove(key.borrow())
    }
}

pub struct TxnMapLockCommitGuard<K, V> {
    keys: TxnLockCommitGuard<BTreeSet<K>>,
    values: Arc<Mutex<BTreeMap<K, Value<V>>>>,
}

impl<K, V> Guard<K, V> for TxnMapLockCommitGuard<K, V>
where
    K: PartialEq,
{
    fn borrow(&self) -> &BTreeSet<K> {
        &self.keys
    }

    fn id(&self) -> &TxnId {
        self.keys.id()
    }

    fn values(&self) -> &Arc<Mutex<BTreeMap<K, Value<V>>>> {
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
pub struct TxnMapLock<K, V> {
    name: Arc<String>,
    keys: TxnLock<BTreeSet<K>>,
    values: Arc<Mutex<BTreeMap<K, Value<V>>>>,
}

impl<K, V> TxnMapLock<K, V> {
    /// Create a new [`TxnMapLock`].
    pub fn new<I: fmt::Display>(name: I) -> Self {
        Self {
            name: Arc::new(name.to_string()),
            keys: TxnLock::new(format!("{} keys", name), BTreeSet::new()),
            values: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }

    /// Get the [`TxnId`] of the last commit to this [`TxnMapLock`].
    pub fn last_commit(&self) -> TxnId {
        self.keys.last_commit()
    }
}

impl<K, V> TxnMapLock<K, V>
where
    K: Ord + Clone,
{
    pub fn with_contents<I, M>(name: I, contents: M) -> Self
    where
        I: fmt::Display,
        M: IntoIterator<Item = (K, V)>,
    {
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

impl<K, V> TxnMapLock<K, V>
where
    K: Clone + PartialEq,
    V: Clone,
{
    /// Lock this map for reading at the given `txn_id`.
    pub async fn read(&self, txn_id: TxnId) -> TCResult<TxnMapLockReadGuard<K, V>> {
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
    pub fn try_read(&self, txn_id: TxnId) -> TCResult<TxnMapLockReadGuard<K, V>> {
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
    pub async fn read_exclusive(
        &self,
        txn_id: TxnId,
    ) -> TCResult<TxnMapLockReadGuardExclusive<K, V>> {
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
    pub async fn write(&self, txn_id: TxnId) -> TCResult<TxnMapLockWriteGuard<K, V>> {
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
impl<K, V> Transact for TxnMapLock<K, V>
where
    K: Ord + PartialEq + Clone + Send + Sync,
    V: Clone + Send + Sync,
{
    type Commit = TxnMapLockCommitGuard<K, V>;

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
