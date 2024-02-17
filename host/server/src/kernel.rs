use async_trait::async_trait;
use log::debug;
use umask::Mode;

use tc_error::*;
use tc_scalar::OpRefType;
use tc_state::CacheBlock;
use tc_transact::public::*;
use tc_transact::{fs, Transact, Transaction, TxnId};
use tc_value::{Link, Value};
use tcgeneric::{label, Label, Map, PathSegment, TCPath};

use crate::cluster::{Class, Cluster, Dir, Schema};
use crate::txn::{Hypothetical, Txn};
use crate::{Authorize, State};

const CLASS: Label = label("class");

pub struct Endpoint<'a> {
    mode: Mode,
    path: &'a [PathSegment],
    handler: Box<dyn Handler<'a, State> + 'a>,
}

impl<'a> Endpoint<'a> {
    pub fn umask(&self) -> Mode {
        self.mode
    }

    pub fn get(self, txn: &'a Txn, key: Value) -> TCResult<GetFuture<State>> {
        let get = self
            .handler
            .get()
            .ok_or_else(|| TCError::method_not_allowed(OpRefType::Get, TCPath::from(self.path)))?;

        if self.mode.may_read() {
            Ok((get)(txn, key))
        } else {
            Err(unauthorized!("read {}", TCPath::from(self.path)))
        }
    }

    pub fn put(self, txn: &'a Txn, key: Value, value: State) -> TCResult<PutFuture> {
        let put = self
            .handler
            .put()
            .ok_or_else(|| TCError::method_not_allowed(OpRefType::Put, TCPath::from(self.path)))?;

        if self.mode.may_write() {
            Ok((put)(txn, key, value))
        } else {
            Err(unauthorized!("read {}", TCPath::from(self.path)))
        }
    }

    pub fn post(self, txn: &'a Txn, params: Map<State>) -> TCResult<PostFuture<State>> {
        let post = self
            .handler
            .post()
            .ok_or_else(|| TCError::method_not_allowed(OpRefType::Post, TCPath::from(self.path)))?;

        if self.mode.may_execute() {
            Ok((post)(txn, params))
        } else {
            Err(unauthorized!("execute {}", TCPath::from(self.path)))
        }
    }

    pub fn delete(self, txn: &'a Txn, key: Value) -> TCResult<DeleteFuture> {
        let delete = self.handler.delete().ok_or_else(|| {
            TCError::method_not_allowed(OpRefType::Delete, TCPath::from(self.path))
        })?;

        if self.mode.may_write() {
            Ok((delete)(txn, key))
        } else {
            Err(unauthorized!("read {}", TCPath::from(self.path)))
        }
    }
}

pub(crate) struct Kernel {
    class: Cluster<Dir<Class>>,
    hypothetical: Cluster<Hypothetical>,
}

impl Kernel {
    pub(crate) fn authorize_claim_and_route<'a>(
        &'a self,
        path: &'a [PathSegment],
        txn: Txn,
    ) -> TCResult<(Txn, Endpoint<'a>)> {
        if path.is_empty() {
            Err(unauthorized!("access to /"))
        } else if path[0] == CLASS {
            auth_claim_route(&self.class, &path[1..], txn)
        } else if path.len() >= 2 && &path[..2] == &Hypothetical::PATH[..] {
            auth_claim_route(&self.hypothetical, &path[2..], txn)
        } else {
            Err(not_found!("{}", TCPath::from(path)))
        }
    }
}

impl Kernel {
    pub(crate) async fn commit(&self, txn_id: TxnId) {
        self.hypothetical.rollback(&txn_id).await;
    }

    pub(crate) async fn finalize(&self, txn_id: TxnId) {
        self.hypothetical.finalize(&txn_id).await;
    }
}

#[async_trait]
impl fs::Persist<CacheBlock> for Kernel {
    type Schema = (Option<Link>, Option<Link>);
    type Txn = Txn;

    async fn create(
        txn_id: TxnId,
        schema: Self::Schema,
        store: fs::Dir<CacheBlock>,
    ) -> TCResult<Self> {
        let (owner, group) = schema;

        let class_dir: fs::Dir<CacheBlock> = store.create_dir(txn_id, CLASS.into()).await?;
        let schema = Schema::new(CLASS, owner.clone(), group.clone());
        let class = fs::Persist::<CacheBlock>::create(txn_id, schema, class_dir).await?;

        let schema = Schema::new(Hypothetical::PATH, owner, group);
        let hypothetical = Cluster::new(schema, Hypothetical::new(), txn_id);

        Ok(Self {
            class,
            hypothetical,
        })
    }

    async fn load(
        txn_id: TxnId,
        schema: Self::Schema,
        store: fs::Dir<CacheBlock>,
    ) -> TCResult<Self> {
        let (owner, group) = schema;

        let class_dir: fs::Dir<CacheBlock> = store.create_dir(txn_id, CLASS.into()).await?;
        let schema = Schema::new(CLASS, owner.clone(), group.clone());
        let class = fs::Persist::<CacheBlock>::load(txn_id, schema, class_dir).await?;

        let schema = Schema::new(Hypothetical::PATH, owner, group);
        let hypothetical = Cluster::new(schema, Hypothetical::new(), txn_id);

        Ok(Self {
            class,
            hypothetical,
        })
    }

    fn dir(&self) -> fs::Inner<CacheBlock> {
        unimplemented!("Kernel::inner")
    }
}

fn auth_claim_route<'a, T>(
    cluster: &'a Cluster<T>,
    path: &'a [PathSegment],
    txn: Txn,
) -> TCResult<(Txn, Endpoint<'a>)>
where
    Cluster<T>: Route<State>,
{
    let txn_id = *txn.id();
    let keyring = cluster.keyring(txn_id)?;
    let resource_mode = cluster.umask(txn_id, path);
    let request_mode = txn.mode(keyring, path);
    let mode = resource_mode & request_mode;

    if mode == Mode::new() {
        return Err(unauthorized!("access to {}", TCPath::from(path)));
    } else {
        debug!("request permissions are {mode}");
    }

    let txn = cluster.claim(txn)?;

    let handler = cluster
        .route(&*path)
        .ok_or_else(|| TCError::not_found(TCPath::from(path)))?;

    let endpoint = Endpoint {
        mode,
        path,
        handler,
    };

    Ok((txn, endpoint))
}
