use std::collections::BTreeMap;
use std::convert::{TryFrom, TryInto};
use std::sync::Arc;

use log::{debug, trace};
use safecast::TryCastInto;
use tokio::sync::RwLock;

use tc_error::*;
use tc_transact::fs::{Dir, DirCreate, Persist};
use tc_transact::Transaction;
use tc_value::{Link, LinkHost, Value};
use tcgeneric::*;

use crate::chain::{Chain, ChainType};
use crate::cluster::ReplicaSet;
use crate::collection::{CollectionBase, CollectionSchema};
use crate::fs;
use crate::object::{InstanceClass, InstanceExt};
use crate::scalar::{OpRef, Refer, Scalar};
use crate::txn::{Actor, Txn, TxnId};

use super::{Cluster, Legacy};

/// Load a cluster from the filesystem, or instantiate a new one.
pub async fn instantiate(
    txn: &Txn,
    host: LinkHost,
    class: InstanceClass,
    data_dir: fs::Dir,
) -> TCResult<InstanceExt<Cluster<Legacy>>> {
    let (link, proto) = class.into_inner();
    let link = link.ok_or_else(|| {
        TCError::unsupported("cluster config must specify a Link to the cluster to host")
    })?;

    let mut chain_schema = Map::new();
    let mut cluster_proto = Map::new();
    let mut classes = Map::new();

    for (id, scalar) in proto.into_iter() {
        debug!("Cluster member: {}", scalar);

        match scalar {
            Scalar::Op(op_def) => {
                let op_def = if op_def.is_write() {
                    // make sure not to replicate ops internal to this OpDef
                    let op_def = op_def.reference_self(link.path());

                    for (id, provider) in op_def.form() {
                        // make sure not to duplicate requests to other clusters
                        if provider.is_inter_service_write(link.path()) {
                            return Err(TCError::unsupported(format!(
                                "replicated op {} may not perform inter-service writes: {}",
                                id, provider
                            )));
                        }
                    }

                    op_def
                } else {
                    // make sure to replicate all write ops internal to this OpDef
                    // by routing them through the kernel
                    op_def.dereference_self(link.path())
                };

                cluster_proto.insert(id, Scalar::Op(op_def));
            }
            Scalar::Ref(tc_ref) => {
                let op_ref = OpRef::try_from(*tc_ref)?;
                match op_ref {
                    OpRef::Get((class, schema)) => {
                        let classpath = TCPathBuf::try_from(class)?;
                        let ct = ChainType::from_path(&classpath)
                            .ok_or_else(|| TCError::bad_request("not a Chain", classpath))?;

                        debug!("an instance of {} with schema {}", ct, schema);
                        let schema = schema.try_cast_into(|scalar| {
                            TCError::bad_request("invalid schema for Collection", scalar)
                        })?;

                        let schema = CollectionSchema::from_scalar(schema)?;

                        chain_schema.insert(id, (ct, schema));
                    }
                    OpRef::Post((extends, proto)) => {
                        let extends = extends.try_into()?;
                        let link = link.clone().append(id.clone());
                        classes.insert(id, InstanceClass::extend(extends, Some(link), proto));
                    }
                    other => {
                        return Err(TCError::bad_request(
                            "expected a Chain or Class but found",
                            other,
                        ))
                    }
                }
            }
            Scalar::Value(Value::Link(extends)) => {
                let link = link.clone().append(id.clone());
                classes.insert(
                    id,
                    InstanceClass::extend(extends, Some(link), Map::default()),
                );
            }
            other => {
                let err_msg = format!("Cluster member must be a Class, Chain (for mutable data), or an OpDef, not {:?}", other);

                return Err(TCError::unsupported(err_msg));
            }
        }
    }

    let txn_id = *txn.id();
    let dir = get_or_create_dir(data_dir, txn_id, link.path()).await?;

    trace!("Cluster::load got write lock on data directory");

    let lead = Link::from((host, link.path().clone()));
    let replica_set = ReplicaSet::new(txn_id, link.clone(), [lead]);

    let mut chains = Map::<Chain<CollectionBase>>::new();
    for (id, (class, schema)) in chain_schema.into_iter() {
        debug!("load chain {} of type {} with schema {}", id, class, schema);

        let store = dir.get_or_create_store(txn_id, id.clone()).await?;
        let chain = Chain::load_or_create(txn_id, (class, schema), store)?;
        trace!("loaded chain {} at {}", id, txn_id);

        chains.insert(id, chain);
    }

    let actor_id = Value::from(Link::default());

    let cluster = Cluster {
        actor: Arc::new(Actor::new(actor_id)),
        led: Arc::new(RwLock::new(BTreeMap::new())),
        replicas: replica_set,
        state: Legacy { chains, classes },
    };

    let class = InstanceClass::new(Some(link), cluster_proto.into());

    Ok(InstanceExt::new(cluster, class))
}

async fn get_or_create_dir(
    data_dir: fs::Dir,
    txn_id: TxnId,
    path: &[PathSegment],
) -> TCResult<fs::Dir> {
    let mut dir = data_dir;
    for name in path {
        dir = {
            let mut dir = dir.write(txn_id).await?;
            dir.get_or_create_dir(name.clone())?
        }
    }

    Ok(dir)
}
