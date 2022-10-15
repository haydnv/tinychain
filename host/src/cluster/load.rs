use std::collections::{HashMap, HashSet};
use std::convert::{TryFrom, TryInto};
use std::sync::Arc;

use log::{debug, trace};
use tokio::sync::RwLock;

use tc_error::*;
use tc_transact::fs::{Dir, DirCreate};
use tc_transact::lock::TxnLock;
use tc_transact::Transaction;
use tc_value::{Link, LinkHost, Value};
use tcgeneric::*;

use crate::chain::{self, Chain, ChainType, Schema};
use crate::fs;
use crate::object::{InstanceClass, InstanceExt};
use crate::scalar::{OpRef, Refer, Scalar};
use crate::txn::{Actor, Txn, TxnId};

use super::Cluster;

/// Load a cluster from the filesystem, or instantiate a new one.
pub async fn instantiate(
    txn: &Txn,
    host: LinkHost,
    class: InstanceClass,
    data_dir: fs::Dir,
) -> TCResult<InstanceExt<Cluster>> {
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
                        let schema = Schema::from_scalar(schema)?;
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
    let mut dir = {
        let dir = get_or_create_dir(data_dir, txn_id, link.path()).await?;
        dir.write(txn_id).await?
    };

    trace!("Cluster::load got write lock on data directory");

    let mut replicas = HashSet::new();
    replicas.insert((host, link.path().clone()).into());

    let mut chains = Map::<Chain>::new();
    for (id, (class, schema)) in chain_schema.into_iter() {
        debug!("load chain {} of type {} with schema {}", id, class, schema);

        let dir = dir.get_or_create_dir(id.clone())?;
        let chain = chain::load(txn, class, schema, dir).await?;
        trace!("loaded chain {}", id);

        chains.insert(id, chain);
    }

    let actor_id = Value::from(Link::default());

    let cluster = Cluster {
        link: link.clone(),
        actor: Arc::new(Actor::new(actor_id)),
        chains,
        classes,
        owned: RwLock::new(HashMap::new()),
        replicas: TxnLock::new(format!("Cluster {} replicas", link), replicas),
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
