use std::marker::PhantomData;

use async_trait::async_trait;
use umask::Mode;

use tc_error::*;
use tc_transact::public::{Handler, Route, StateInstance};
use tc_transact::{fs, Transact, TxnId};
use tc_value::Link;
use tcgeneric::{PathSegment, TCPath, ThreadSafe};

use crate::cluster::{Cluster, Schema};
use crate::txn::{Hypothetical, Txn};

pub(crate) struct Kernel<State, FE> {
    hypothetical: Cluster<Hypothetical>,
    state: PhantomData<(State, FE)>,
}

impl<State, FE> Kernel<State, FE> {
    pub fn authorize_claim_and_route<'a>(
        &'a self,
        mode: Mode,
        path: &'a [PathSegment],
        txn: Txn<State, FE>,
    ) -> TCResult<(Txn<State, FE>, Box<dyn Handler<State> + 'a>)>
    where
        FE: ThreadSafe + Clone,
        State: StateInstance<FE = FE, Txn = Txn<State, FE>>,
    {
        if path.is_empty() {
            Err(unauthorized!("access to /"))
        } else if path.len() >= 2 && &path[..2] == &Hypothetical::PATH[..] {
            if self.hypothetical.authorization(path, &txn) & mode == mode {
                let txn = maybe_claim_txn(&self.hypothetical, txn);

                let handler = self
                    .hypothetical
                    .route(path)
                    .ok_or_else(|| TCError::not_found(TCPath::from(&path[2..])))?;

                Ok((txn, handler))
            } else {
                Err(unauthorized!("access to {}", TCPath::from(path)))
            }
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

    async fn create(_txn_id: TxnId, schema: Self::Schema, _store: fs::Dir<FE>) -> TCResult<Self> {
        let (owner, group) = schema;
        let schema = Schema::new(Hypothetical::PATH, owner, group);

        Ok(Self {
            hypothetical: Cluster::new(schema, Hypothetical::new()),
            state: PhantomData,
        })
    }

    async fn load(_txn_id: TxnId, schema: Self::Schema, _store: fs::Dir<FE>) -> TCResult<Self> {
        let (owner, group) = schema;
        let schema = Schema::new(Hypothetical::PATH, owner, group);

        Ok(Self {
            hypothetical: Cluster::new(schema, Hypothetical::new()),
            state: PhantomData,
        })
    }

    fn dir(&self) -> fs::Inner<FE> {
        unimplemented!("Kernel::inner")
    }
}

#[inline]
fn maybe_claim_txn<State, FE, T>(cluster: &Cluster<T>, txn: Txn<State, FE>) -> Txn<State, FE> {
    if txn.owner().is_none() || txn.leader(cluster.path()).is_none() {
        txn.claim(cluster.public_key(), cluster.path())
    } else {
        txn
    }
}
