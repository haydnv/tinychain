use std::marker::PhantomData;

use async_trait::async_trait;
use log::debug;
use umask::Mode;

use tc_error::*;
use tc_scalar::{OpRefType, Refer, Scalar};
use tc_transact::public::{Handler, PostFuture, Route, StateInstance};
use tc_transact::{fs, Transact, Transaction, TxnId};
use tc_value::Link;
use tcgeneric::{Map, PathSegment, TCPath, ThreadSafe};

use crate::cluster::{Cluster, Schema};
use crate::txn::{Hypothetical, Txn};
use crate::Authorize;

pub struct Endpoint<'a, State> {
    mode: Mode,
    path: &'a [PathSegment],
    handler: Box<dyn Handler<'a, State> + 'a>,
}

impl<'a, State> Endpoint<'a, State> {
    pub fn umask(&self) -> Mode {
        self.mode
    }

    pub fn post<FE>(
        self,
        txn: &'a Txn<State, FE>,
        params: Map<State>,
    ) -> TCResult<PostFuture<State>>
    where
        State: StateInstance<FE = FE, Txn = Txn<State, FE>>,
    {
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
}

pub(crate) struct Kernel<State, FE> {
    hypothetical: Cluster<Hypothetical>,
    state: PhantomData<(State, FE)>,
}

impl<State, FE> Kernel<State, FE> {
    pub(crate) fn authorize_claim_and_route<'a>(
        &'a self,
        path: &'a [PathSegment],
        txn: Txn<State, FE>,
    ) -> TCResult<(Txn<State, FE>, Endpoint<'a, State>)>
    where
        FE: ThreadSafe + Clone,
        State: StateInstance<FE = FE, Txn = Txn<State, FE>> + Refer<State> + From<Scalar>,
        Scalar: TryFrom<State, Error = TCError>,
    {
        if path.is_empty() {
            Err(unauthorized!("access to /"))
        } else if path.len() >= 2 && &path[..2] == &Hypothetical::PATH[..] {
            auth_claim_route(&self.hypothetical, &path[2..], txn)
        } else {
            Err(not_found!("{}", TCPath::from(path)))
        }
    }
}

impl<State, FE> Kernel<State, FE> {
    pub(crate) async fn commit(&self, txn_id: TxnId) {
        self.hypothetical.rollback(&txn_id).await;
    }

    pub(crate) async fn finalize(&self, txn_id: TxnId) {
        self.hypothetical.finalize(&txn_id).await;
    }
}

#[async_trait]
impl<State, FE> fs::Persist<FE> for Kernel<State, FE>
where
    State: ThreadSafe,
    FE: ThreadSafe + Clone,
{
    type Schema = (Option<Link>, Option<Link>);
    type Txn = Txn<State, FE>;

    async fn create(txn_id: TxnId, schema: Self::Schema, _store: fs::Dir<FE>) -> TCResult<Self> {
        let (owner, group) = schema;
        let schema = Schema::new(Hypothetical::PATH, owner, group);

        Ok(Self {
            hypothetical: Cluster::new(schema, Hypothetical::new(), txn_id),
            state: PhantomData,
        })
    }

    async fn load(txn_id: TxnId, schema: Self::Schema, _store: fs::Dir<FE>) -> TCResult<Self> {
        let (owner, group) = schema;
        let schema = Schema::new(Hypothetical::PATH, owner, group);

        Ok(Self {
            hypothetical: Cluster::new(schema, Hypothetical::new(), txn_id),
            state: PhantomData,
        })
    }

    fn dir(&self) -> fs::Inner<FE> {
        unimplemented!("Kernel::inner")
    }
}

fn auth_claim_route<'a, State, FE, T>(
    cluster: &'a Cluster<T>,
    path: &'a [PathSegment],
    txn: Txn<State, FE>,
) -> TCResult<(Txn<State, FE>, Endpoint<'a, State>)>
where
    State: StateInstance<FE = FE, Txn = Txn<State, FE>>,
    FE: ThreadSafe + Clone,
    Cluster<T>: Route<State>,
    Scalar: TryFrom<State, Error = TCError>,
{
    let keyring = cluster.keyring(*txn.id())?;
    let resource_mode = cluster.umask(path);
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
        .ok_or_else(|| TCError::not_found(TCPath::from(&path[2..])))?;

    let endpoint = Endpoint {
        mode,
        path,
        handler,
    };

    Ok((txn, endpoint))
}
