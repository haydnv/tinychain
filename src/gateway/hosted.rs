use std::collections::HashMap;

use crate::cluster::Cluster;
use crate::scalar::value::link::{PathSegment, TCPath};

#[derive(Clone)]
struct HostedNode {
    children: HashMap<PathSegment, HostedNode>,
}

pub struct Hosted {
    root: HostedNode,
    hosted: HashMap<TCPath, Cluster>,
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

    pub fn get(&self, path: &TCPath) -> Option<(TCPath, Cluster)> {
        println!("checking for hosted cluster {}", path);
        let mut node = &self.root;
        let mut found_path = TCPath::default();
        for segment in path.clone() {
            println!("checking for segment {}", segment);

            if let Some(child) = node.children.get(&segment) {
                found_path.push(segment);
                node = child;
                println!("found segment: {}", found_path);
            } else if let Some(cluster) = self.hosted.get(&found_path) {
                return Some((path.from_path(&found_path).unwrap(), cluster.clone()));
            } else {
                println!("couldn't find {}", segment);
                return None;
            }
        }

        if let Some(cluster) = self.hosted.get(&found_path) {
            Some((path.from_path(&found_path).unwrap(), cluster.clone()))
        } else {
            None
        }
    }

    pub fn push(&mut self, path: TCPath, cluster: Cluster) -> Option<Cluster> {
        let mut node = &mut self.root;
        for segment in path.clone() {
            node = node.children.entry(segment).or_insert(HostedNode {
                children: HashMap::new(),
            });
        }

        println!("Hosted directory: {}", path);
        self.hosted.insert(path, cluster)
    }
}
