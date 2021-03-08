use std::collections::{HashMap, HashSet};
use std::convert::{TryFrom, TryInto};
use std::sync::Arc;

use futures::future::TryFutureExt;
use log::debug;
use safecast::TryCastInto;
use uplock::RwLock;

use tc_error::*;
use tc_transact::fs::Persist;
use tc_transact::lock::TxnLock;
use tcgeneric::*;

use crate::chain::{Chain, ChainType, SyncChain};
use crate::fs;
use crate::object::{InstanceClass, InstanceExt};
use crate::scalar::{Link, OpRef, Scalar, Value};
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
    let mut classes = HashMap::new();

    for (id, scalar) in proto.into_iter() {
        debug!("Cluster member: {}", scalar);

        match scalar {
            Scalar::Ref(tc_ref) => {
                let op_ref = OpRef::try_from(*tc_ref)?;
                match op_ref {
                    OpRef::Get((subject, schema)) => {
                        let classpath = TCPathBuf::try_from(subject)?;
                        let ct = ChainType::from_path(&classpath)
                            .ok_or_else(|| TCError::bad_request("not a Chain", classpath))?;

                        let schema: Value = schema.try_into()?;
                        chain_schema.insert(id, (ct, schema));
                    }
                    OpRef::Post((extends, proto)) => {
                        let extends = extends.try_into()?;
                        classes.insert(id, InstanceClass::new(Some(extends), proto));
                    }
                    other => return Err(TCError::bad_request("expected a Chain but found", other)),
                }
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
        classes: classes.into(),
        confirmed: RwLock::new(txn_id),
        owned: RwLock::new(HashMap::new()),
        installed: TxnLock::new(
            format!("Cluster {} installed deps", path),
            HashMap::new().into(),
        ),
        replicas: TxnLock::new(format!("Cluster {} replicas", path), HashSet::new().into()),
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
