use std::collections::HashMap;
use std::hash::Hash;
use std::iter::FromIterator;
use std::sync::RwLock;

pub struct Map<K: Clone + Eq + Hash + Send + Sync, V: Send + Sync> {
    map: RwLock<HashMap<K, V>>,
}

impl<K: Clone + Eq + Hash + Send + Sync, V: Clone + Send + Sync> Map<K, V> {
    pub fn new() -> Map<K, V> {
        Map {
            map: RwLock::new(HashMap::new()),
        }
    }

    pub fn get(&self, key: &K) -> Option<V> {
        match self.map.read().unwrap().get(key) {
            Some(val) => Some(val.clone()),
            None => None,
        }
    }

    pub fn insert(&self, key: K, value: V) -> Option<V> {
        self.map.write().unwrap().insert(key, value)
    }
}

impl<K: Clone + Eq + Hash + Send + Sync, V: Send + Sync> FromIterator<(K, V)> for Map<K, V> {
    fn from_iter<T: IntoIterator<Item = (K, V)>>(i: T) -> Map<K, V> {
        let mut map: HashMap<K, V> = HashMap::new();
        for (k, v) in i {
            map.insert(k, v);
        }

        Map {
            map: RwLock::new(map),
        }
    }
}
