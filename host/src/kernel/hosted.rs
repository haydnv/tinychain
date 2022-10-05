use std::collections::HashMap;
use std::iter::FromIterator;

use log::{debug, info};

use tcgeneric::{label, Label, PathSegment, TCPath, TCPathBuf};

use crate::cluster::{Cluster, Legacy};
use crate::object::InstanceExt;

const RESERVED: [Label; 57] = [
    label("actor"),
    label("actors"),
    label("admin"),
    label("administration"),
    label("administrator"),
    label("auth"),
    label("authenticate"),
    label("authentication"),
    label("authorize"),
    label("authorized"),
    label("authorization"),
    label("boot"),
    label("channel"),
    label("const"),
    label("crypt"),
    label("crypto"),
    label("decrypt"),
    label("decryption"),
    label("decrypted"),
    label("dev"),
    label("encrypt"),
    label("encryption"),
    label("encrypted"),
    label("error"),
    label("https"),
    label("host"),
    label("internal"),
    label("kernel"),
    label("link"),
    label("market"),
    label("null"),
    label("official"),
    label("op"),
    label("ops"),
    label("operations"),
    label("operator"),
    label("recycle"),
    label("recycling"),
    label("secure"),
    label("security"),
    label("shortcut"),
    label("sink"),
    label("socket"),
    label("ssl"),
    label("state"),
    label("symbolic"),
    label("symlink"),
    label("sys"),
    label("system"),
    label("tls"),
    label("token"),
    label("transact"),
    label("transaction"),
    label("trash"),
    label("user"),
    label("users"),
    label("waste"),
];

#[derive(Clone)]
struct HostedNode {
    children: HashMap<PathSegment, HostedNode>,
}

pub struct Hosted {
    root: HostedNode,
    hosted: HashMap<TCPathBuf, InstanceExt<Legacy>>,
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

    pub fn clusters(&self) -> impl Iterator<Item = &InstanceExt<Legacy>> {
        self.hosted.values()
    }

    pub fn get<'a>(
        &self,
        path: &'a [PathSegment],
    ) -> Option<(&'a [PathSegment], &InstanceExt<Legacy>)> {
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

    fn push(&mut self, cluster: InstanceExt<Legacy>) {
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

impl FromIterator<InstanceExt<Legacy>> for Hosted {
    fn from_iter<I: IntoIterator<Item = InstanceExt<Legacy>>>(iter: I) -> Self {
        let mut hosted = Hosted::new();

        for cluster in iter.into_iter() {
            hosted.push(cluster);
        }

        hosted
    }
}
