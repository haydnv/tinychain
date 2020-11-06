use std::collections::HashMap;

use log::{debug, info};

use crate::cluster::Cluster;
use crate::scalar::value::link::{PathSegment, TCPath, TCPathBuf};

#[derive(Clone)]
struct HostedNode {
    children: HashMap<PathSegment, HostedNode>,
}

pub struct Hosted {
    root: HostedNode,
    hosted: HashMap<TCPathBuf, Cluster>,
}

impl Hosted {
    pub fn new() -> Hosted {
        Hosted {
            root: HostedNode {
                children: HashMap::new(),
            },
            hosted: HashMap::new(),
        }
    }

    pub fn get<'a>(&self, path: &'a [PathSegment]) -> Option<(&'a [PathSegment], Cluster)> {
        debug!("checking for hosted cluster {}", TCPath::from(path));

        let mut node = &self.root;
        let mut found_path = &path[0..0];
        for i in 0..path.len() {
            if let Some(child) = node.children.get(&path[i]) {
                found_path = &path[..i];
                node = child;
            } else if let Some(cluster) = self.hosted.get(found_path) {
                return Some((&path[found_path.len()..], cluster.clone()));
            } else {
                return None;
            }
        }

        if let Some(cluster) = self.hosted.get(found_path) {
            Some((&path[found_path.len()..], cluster.clone()))
        } else {
            None
        }
    }

    pub fn push(&mut self, path: TCPathBuf, cluster: Cluster) -> Option<Cluster> {
        let mut node = &mut self.root;
        for segment in path.clone() {
            node = node.children.entry(segment).or_insert(HostedNode {
                children: HashMap::new(),
            });
        }

        info!("Hosted directory: {}", path);
        self.hosted.insert(path, cluster)
    }
}
