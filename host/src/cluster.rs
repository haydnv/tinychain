use std::collections::HashMap;
use std::fmt;

use destream::{de, Decoder, FromStream};
use futures::{future, stream, TryFutureExt};

use error::*;
use generic::*;

use crate::chain::Chain;
use crate::fs;
use crate::object::{InstanceClass, InstanceExt};
use crate::scalar::Scalar;
use crate::state::State;
use crate::txn::Txn;

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
    chains: Map<Chain>,
}

impl Cluster {
    pub async fn load(
        data_dir: fs::Dir,
        txn: Txn,
        path: TCPathBuf,
        config: Vec<u8>,
    ) -> TCResult<InstanceExt<Self>> {
        let decoder = destream_json::de::Decoder::from(stream::once(future::ready(Ok(config))));
        from_stream(data_dir, txn, path, decoder)
            .map_err(|e| TCError::bad_request("error decoding cluster definition", e))
            .await
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

async fn from_stream<D: Decoder>(
    _dir: fs::Dir,
    txn: Txn,
    path: TCPathBuf,
    mut decoder: D,
) -> Result<InstanceExt<Cluster>, D::Error> {
    let state = State::from_stream(txn, &mut decoder).await?;
    let members = if let State::Map(members) = state {
        members
    } else {
        return Err(de::Error::invalid_value(state, "a Cluster definition"));
    };

    let chains = HashMap::new();
    let mut proto = HashMap::new();

    for (id, state) in members.into_iter() {
        match state {
            State::Chain(_chain) => {
                todo!();
            }
            State::Scalar(Scalar::Op(op)) => {
                proto.insert(id, Scalar::Op(op));
            }
            other => return Err(de::Error::invalid_value(other, "a Chain or Op")),
        }
    }

    let class = InstanceClass::new(Some(path.clone().into()), proto.into());
    let cluster = Cluster {
        path,
        chains: chains.into(),
    };
    Ok(InstanceExt::new(cluster, class))
}
