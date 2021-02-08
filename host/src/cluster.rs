use std::collections::HashMap;
use std::convert::TryInto;
use std::fmt;

use async_trait::async_trait;
use destream::{de, Decoder, FromStream, MapAccess};
use futures::{future, stream, TryFutureExt};

use error::*;
use generic::*;
use transact::fs::Dir;
use transact::TxnId;

use crate::chain::sync::SyncChain;
use crate::chain::{Chain, ChainBlock, ChainType};
use crate::fs;
use crate::object::{InstanceClass, InstanceExt};
use crate::scalar::{OpDef, OpDefType, Scalar, ScalarType};
use crate::state::StateType;

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
        txn_id: TxnId,
        path: TCPathBuf,
        config: Vec<u8>,
    ) -> TCResult<InstanceExt<Self>> {
        let decoder = destream_json::de::Decoder::from(stream::once(future::ready(Ok(config))));
        from_stream(data_dir, txn_id, path, decoder)
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

#[derive(Clone)]
enum ClusterState {
    Chain(ChainType, Scalar),
    Op(OpDef),
}

#[async_trait]
impl de::FromStream for ClusterState {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Self, D::Error> {
        decoder.decode_map(Visitor).await
    }
}

struct Visitor;

#[async_trait]
impl de::Visitor for Visitor {
    type Value = ClusterState;

    fn expecting() -> &'static str {
        "a Chain or Op definition"
    }

    async fn visit_map<A: MapAccess>(self, mut map: A) -> Result<Self::Value, A::Error> {
        if let Some(path) = map.next_key::<TCPathBuf>(()).await? {
            if let Some(class) = StateType::from_path(&path) {
                match class {
                    StateType::Chain(ct) => {
                        let schema = map.next_value(()).await?;
                        Ok(ClusterState::Chain(ct, schema))
                    }
                    StateType::Scalar(ScalarType::Op(odt)) => {
                        let op_def = match odt {
                            OpDefType::Get => map.next_value(()).map_ok(OpDef::Get).await,
                            OpDefType::Put => map.next_value(()).map_ok(OpDef::Put).await,
                            OpDefType::Post => map.next_value(()).map_ok(OpDef::Post).await,
                            OpDefType::Delete => map.next_value(()).map_ok(OpDef::Delete).await,
                        }?;

                        Ok(ClusterState::Op(op_def))
                    }
                    other => Err(de::Error::invalid_value(other, Self::expecting())),
                }
            } else {
                Err(de::Error::invalid_value(path, Self::expecting()))
            }
        } else {
            Err(de::Error::invalid_length(0, Self::expecting()))
        }
    }
}

async fn from_stream<D: Decoder>(
    dir: fs::Dir,
    txn_id: TxnId,
    path: TCPathBuf,
    mut decoder: D,
) -> Result<InstanceExt<Cluster>, D::Error> {
    let members = Map::<ClusterState>::from_stream((), &mut decoder).await?;

    let mut chains = HashMap::new();
    let mut proto = HashMap::new();

    for (id, state) in members.into_iter() {
        match state {
            ClusterState::Chain(ct, schema) => {
                let file: fs::File<ChainBlock> = if let Some(file) = dir
                    .get_file(&txn_id, &id)
                    .map_err(de::Error::custom)
                    .await?
                {
                    file.try_into().map_err(de::Error::custom)?
                } else {
                    let file = dir
                        .create_file(txn_id, id.clone(), StateType::Chain(ct))
                        .map_err(de::Error::custom)
                        .await?;

                    file.try_into().map_err(de::Error::custom)?
                };

                let chain = match ct {
                    ChainType::Sync => {
                        SyncChain::load(file, schema)
                            .map_ok(Chain::Sync)
                            .map_err(de::Error::custom)
                            .await?
                    }
                };

                chains.insert(id, chain);
            }
            ClusterState::Op(op_def) => {
                proto.insert(id, Scalar::Op(op_def));
            }
        }
    }

    let class = InstanceClass::new(Some(path.clone().into()), proto.into());
    let cluster = Cluster {
        path,
        chains: chains.into(),
    };
    Ok(InstanceExt::new(cluster, class))
}
