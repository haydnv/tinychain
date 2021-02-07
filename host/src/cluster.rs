use std::fmt;
use std::sync::Arc;

use bytes::Bytes;
use destream::de;
use destream_json::de::Decoder;
use futures::{future, stream};

use error::*;
use generic::*;
use transact::fs;

use crate::object::InstanceExt;
use crate::txn::{FileEntry, TxnId};

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
        txn_id: TxnId,
        data_dir: Arc<fs::Dir<FileEntry>>,
        path: TCPathBuf,
        config: Bytes,
    ) -> TCResult<InstanceExt<Self>> {
        if path.is_empty() {
            return Err(TCError::unsupported("Cannot host a cluster at /"));
        }

        let dir = data_dir
            .get_dir(&txn_id, &path)
            .await?
            .ok_or_else(|| TCError::not_found(&path))?;

        let decoder = Decoder::from(stream::once(future::ready(Ok(config.to_vec()))));

        from_stream(dir, decoder)
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

fn from_stream<D: de::Decoder>(
    _dir: Arc<fs::Dir<FileEntry>>,
    _decoder: D,
) -> TCResult<InstanceExt<Cluster>> {
    Err(TCError::not_implemented("cluster::from_stream"))
}
