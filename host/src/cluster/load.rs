use std::collections::HashMap;
use std::convert::TryInto;
use std::sync::Arc;

use futures::future::TryFutureExt;
use safecast::TryCastInto;
use uplock::RwLock;

use tc_error::*;
use tc_transact::fs::Persist;
use tcgeneric::*;

use crate::chain::{Chain, ChainType, SyncChain};
use crate::fs;
use crate::object::{InstanceClass, InstanceExt};
use crate::scalar::{Link, OpRef, Scalar, TCRef, Value};
use crate::txn::{Actor, TxnId};

use super::Cluster;

/// Load a cluster from the filesystem, or instantiate a new one.
pub async fn instantiate(
    class: InstanceClass,
    data_dir: fs::Dir,
    txn_id: TxnId,
) -> TCResult<InstanceExt<Cluster>> {
    let (path, proto) = class.into_inner();
    let path = path.ok_or_else(|| {
        TCError::unsupported("cluster config must specify the path of the cluster to host")
    })?;

    let path = path.into_path();

    let mut chain_schema = HashMap::new();
    let mut cluster_proto = HashMap::new();
    for (id, scalar) in proto.into_iter() {
        match scalar {
            Scalar::Ref(tc_ref) => {
                let (ct, schema) = if let TCRef::Op(OpRef::Get((path, schema))) = *tc_ref {
                    let path: TCPathBuf = path.try_into()?;
                    let schema: Value = schema.try_into()?;

                    if let Some(ct) = ChainType::from_path(&path) {
                        (ct, schema)
                    } else {
                        return Err(TCError::bad_request(
                            "Cluster requires its mutable data to be wrapped in a chain, not",
                            path,
                        ));
                    }
                } else {
                    return Err(TCError::bad_request("expected a Chain but found", tc_ref));
                };

                chain_schema.insert(id, (ct, schema));
            }
            Scalar::Op(op_def) => {
                cluster_proto.insert(id, Scalar::Op(op_def));
            }
            other => {
                return Err(TCError::bad_request(
                    "Cluster member must be a Chain (for mutable data), or an immutable OpDef, not",
                    other,
                ))
            }
        }
    }

    let dir = get_or_create_dir(data_dir, txn_id, &path).await?;

    let mut chains = HashMap::<Id, Chain>::new();
    for (id, (class, schema)) in chain_schema.into_iter() {
        let dir = dir.get_or_create_dir(txn_id, id.clone()).await?;
        let chain = match class {
            ChainType::Sync => {
                let schema =
                    schema.try_cast_into(|v| TCError::bad_request("invalid Chain schema", v))?;

                SyncChain::load(schema, dir, txn_id)
                    .map_ok(Chain::Sync)
                    .await?
            }
        };

        chains.insert(id, chain);
    }

    let actor_id = Value::from(Link::default());
    let cluster = Cluster {
        actor: Arc::new(Actor::new(actor_id)),
        path: path.clone(),
        chains: chains.into(),
        confirmed: RwLock::new(txn_id),
        owned: RwLock::new(HashMap::new()),
    };

    let class = InstanceClass::new(Some(path.into()), cluster_proto.into());

    Ok(InstanceExt::new(cluster, class))
}

async fn get_or_create_dir(
    data_dir: fs::Dir,
    txn_id: TxnId,
    path: &[PathSegment],
) -> TCResult<fs::Dir> {
    let mut dir = data_dir;
    for name in path {
        dir = dir.get_or_create_dir(txn_id, name.clone()).await?;
    }

    Ok(dir)
}
