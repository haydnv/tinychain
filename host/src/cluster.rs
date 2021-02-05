use std::fmt;

use futures_locks::RwLockReadGuard;

use error::*;
use generic::{Class, Instance, PathSegment, TCPathBuf};
use transact::fs;

use crate::object::InstanceExt;

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
        _data_dir: RwLockReadGuard<fs::HostDir>,
        _path: TCPathBuf,
    ) -> TCResult<InstanceExt<Self>> {
        unimplemented!()
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
