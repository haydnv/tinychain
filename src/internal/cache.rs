use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hash;
use std::iter::FromIterator;
use std::sync::RwLock;

pub struct Map<K: Clone + Eq + Hash + Send + Sync, V: Send + Sync> {
    map: RwLock<Single<HashMap<K, V>>>,
}

impl<K: Clone + Eq + Hash + Send + Sync, V: Clone + Send + Sync> Map<K, V> {
    pub fn new() -> Map<K, V> {
        Map {
            map: RwLock::new(Single::new(Some(HashMap::new()))),
        }
    }

    pub fn drain(&self) -> HashMap<K, V> {
        self.map
            .write()
            .unwrap()
            .replace(Some(HashMap::new()))
            .unwrap()
    }

    pub fn extend<I: Iterator<Item = (K, V)>>(&self, i: I) {
        let mut lock = self.map.write().unwrap();
        let mut map = lock.replace(None).unwrap();
        map.extend(i);
        lock.replace(Some(map));
    }

    pub fn get(&self, key: &K) -> Option<V> {
        match self.map.read().unwrap().read().unwrap().get(key) {
            Some(val) => Some(val.clone()),
            None => None,
        }
    }

    pub fn insert(&self, key: K, value: V) -> Option<V> {
        let mut lock = self.map.write().unwrap();
        let mut map = lock.replace(None).unwrap();
        let result = map.insert(key, value);
        lock.replace(Some(map));
        result
    }

    pub fn keys(&self) -> HashSet<K> {
        let lock = self.map.read().unwrap();
        let keys = lock.read().unwrap().keys();
        keys.cloned().collect()
    }
}

impl<K: Clone + Eq + Hash + Send + Sync, V: Send + Sync> FromIterator<(K, V)> for Map<K, V> {
    fn from_iter<T: IntoIterator<Item = (K, V)>>(i: T) -> Map<K, V> {
        let mut map: HashMap<K, V> = HashMap::new();
        for (k, v) in i {
            map.insert(k, v);
        }
        Map {
            map: RwLock::new(Single::new(Some(map))),
        }
    }
}

#[derive(Debug)]
pub struct Deque<V> {
    deque: RwLock<VecDeque<V>>,
}

impl<V> Deque<V> {
    pub fn new() -> Deque<V> {
        Deque {
            deque: RwLock::new(VecDeque::new()),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.deque.read().unwrap().is_empty()
    }

    pub fn pop_front(&self) -> Option<V> {
        self.deque.write().unwrap().pop_front()
    }

    pub fn push_back(&self, item: V) {
        self.deque.write().unwrap().push_back(item)
    }
}

pub struct Single<T: Send + Sync> {
    item: Vec<T>,
}

impl<T: Send + Sync> Single<T> {
    pub fn new(item: Option<T>) -> Single<T> {
        let mut v = Vec::with_capacity(1);

        if let Some(item) = item {
            v.push(item);
        }

        Single { item: v }
    }

    pub fn read(&self) -> Option<&T> {
        self.item.last()
    }

    pub fn replace(&mut self, new_item: Option<T>) -> Option<T> {
        let item = self.item.pop();
        if let Some(new_item) = new_item {
            self.item.push(new_item);
        }

        item
    }
}
