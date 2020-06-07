use std::collections::HashMap;
use std::sync::Arc;

use crate::cluster::Cluster;
use crate::value::link::{PathSegment, TCPath};

#[derive(Clone)]
struct HostedNode {
    children: HashMap<PathSegment, HostedNode>,
}

#[derive(Clone)] // TODO: remove Clone trait
pub struct Hosted {
    root: HostedNode,
    hosted: HashMap<TCPath, Arc<Cluster>>,
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

    pub fn get(&self, path: &TCPath) -> Option<(TCPath, Arc<Cluster>)> {
        println!("checking for hosted cluster {}", path);
        let mut node = &self.root;
        let mut found_path = TCPath::default();
        for segment in path.clone() {
            if let Some(child) = node.children.get(&segment) {
                found_path.push(segment);
                node = child;
                println!("found {}", found_path);
            } else if found_path != TCPath::default() {
                return Some((
                    path.from_path(&found_path).unwrap(),
                    self.hosted.get(&found_path).unwrap().clone(),
                ));
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

    #[allow(dead_code)]
    pub fn push(&mut self, path: TCPath, cluster: Arc<Cluster>) -> Option<Arc<Cluster>> {
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
