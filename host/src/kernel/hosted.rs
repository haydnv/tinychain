use std::collections::HashMap;
use std::iter::FromIterator;

use log::{debug, info};

use tcgeneric::{label, Label, PathSegment, TCPath, TCPathBuf};

use crate::cluster::Cluster;
use crate::object::InstanceExt;

const RESERVED: [Label; 3] = [label("error"), label("state"), label("transact")];

#[derive(Clone)]
struct HostedNode {
    children: HashMap<PathSegment, HostedNode>,
}

pub struct Hosted {
    root: HostedNode,
    hosted: HashMap<TCPathBuf, InstanceExt<Cluster>>,
}

impl Hosted {
    fn new() -> Hosted {
        Hosted {
            root: HostedNode {
                children: HashMap::new(),
            },
            hosted: HashMap::new(),
        }
    }

    pub fn get<'a>(
        &self,
        path: &'a [PathSegment],
    ) -> Option<(&'a [PathSegment], &InstanceExt<Cluster>)> {
        debug!("checking for hosted cluster {}", TCPath::from(path));

        let mut node = &self.root;
        let mut found_path = &path[0..0];
        for i in 0..path.len() {
            if let Some(child) = node.children.get(&path[i]) {
                found_path = &path[..i + 1];
                node = child;
            } else if let Some(cluster) = self.hosted.get(found_path) {
                return Some((&path[found_path.len()..], cluster));
            } else {
                return None;
            }
        }

        if let Some(cluster) = self.hosted.get(found_path) {
            Some((&path[found_path.len()..], cluster))
        } else {
            None
        }
    }

    fn push(&mut self, cluster: InstanceExt<Cluster>) {
        if cluster.path().is_empty() {
            panic!("Cannot host a cluster at /");
        } else {
            for id in &RESERVED {
                if &cluster.path()[0] == id {
                    panic!("Cannot host a cluster at reserved path /{}", id);
                }
            }
        }

        let mut node = &mut self.root;
        for segment in cluster.path().iter().cloned() {
            node = node.children.entry(segment).or_insert(HostedNode {
                children: HashMap::new(),
            });
        }

        info!("Hosted {}", cluster);
        self.hosted.insert(cluster.path().to_vec().into(), cluster);
    }
}

impl FromIterator<InstanceExt<Cluster>> for Hosted {
    fn from_iter<I: IntoIterator<Item = InstanceExt<Cluster>>>(iter: I) -> Self {
        let mut hosted = Hosted::new();

        for cluster in iter.into_iter() {
            hosted.push(cluster);
        }

        hosted
    }
}
