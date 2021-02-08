use std::fmt;

use bytes::Bytes;

use error::*;
use generic::*;

use crate::fs::Root;
use crate::object::InstanceExt;

pub const PATH: Label = label("cluster");

pub struct ClusterType;

impl Class for ClusterType {
    type Instance = Cluster;
}

impl fmt::Display for ClusterType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("Cluster")
    }
}

#[derive(Clone)]
pub struct Cluster {
    path: TCPathBuf,
}

impl Cluster {
    pub async fn load(
        _data_dir: Root,
        _path: TCPathBuf,
        _config: Bytes,
    ) -> TCResult<InstanceExt<Self>> {
        Err(TCError::not_implemented("Cluster::load"))
    }

    pub fn path(&'_ self) -> &'_ [PathSegment] {
        &self.path
    }
}

impl Instance for Cluster {
    type Class = ClusterType;

    fn class(&self) -> Self::Class {
        ClusterType
    }
}

impl fmt::Display for Cluster {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Cluster at {}", self.path)
    }
}
