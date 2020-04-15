use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, RwLock};

#[derive(Debug)]
pub struct Map<K: Eq + Hash, V: Hash> {
    map: RwLock<HashMap<K, Arc<V>>>,
}

impl<K: Eq + Hash, V: Hash> Map<K, V> {
    pub fn new() -> Map<K, V> {
        Map {
            map: RwLock::new(HashMap::new()),
        }
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.map.read().unwrap().contains_key(key)
    }

    pub fn get(&self, key: &K) -> Option<Arc<V>> {
        match self.map.read().unwrap().get(key) {
            Some(state) => Some(state.clone()),
            None => None,
        }
    }

    pub fn insert(&self, key: K, value: Arc<V>) -> Option<Arc<V>> {
        self.map.write().unwrap().insert(key, value)
    }

    pub fn remove(&self, key: &K) -> Option<Arc<V>> {
        self.map.write().unwrap().remove(key)
    }
}

#[derive(Debug)]
pub struct Queue<V> {
    queue: RwLock<Vec<Arc<V>>>,
}

impl<V> Queue<V> {
    pub fn new() -> Arc<Queue<V>> {
        Arc::new(Queue {
            queue: RwLock::new(vec![]),
        })
    }

    pub fn with_capacity(i: usize) -> Arc<Queue<V>> {
        Arc::new(Queue {
            queue: RwLock::new(Vec::with_capacity(i)),
        })
    }

    pub fn is_empty(self: &Arc<Self>) -> bool {
        self.queue.read().unwrap().is_empty()
    }

    pub fn last(self: &Arc<Self>) -> Option<Arc<V>> {
        if let Some(v) = self.queue.read().unwrap().last() {
            Some(v.clone())
        } else {
            None
        }
    }

    pub fn push(self: &Arc<Self>, item: V) {
        self.queue.write().unwrap().push(Arc::new(item))
    }

    pub fn pop(self: &Arc<Self>) -> Option<Arc<V>> {
        self.queue.write().unwrap().pop()
    }

    pub fn reverse(self: &Arc<Self>) {
        self.queue.write().unwrap().reverse()
    }
}
