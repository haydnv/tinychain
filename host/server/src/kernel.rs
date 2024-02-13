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

pub(crate) struct Kernel<FE> {
    hypothetical: Cluster<Hypothetical>,
    file: PhantomData<FE>,
}

impl<FE> Kernel<FE> {
    pub fn authorize_and_route<'a, State>(
        &'a self,
        mode: Mode,
        path: &'a [PathSegment],
        txn: &Txn<FE>,
    ) -> TCResult<Box<dyn Handler<State> + 'a>>
    where
        FE: ThreadSafe + Clone,
        State: StateInstance<FE = FE, Txn = Txn<FE>>,
    {
        if path.is_empty() {
            Err(unauthorized!("access to /"))
        } else if path.len() >= 2 && &path[..2] == &Hypothetical::PATH[..] {
            if self.hypothetical.authorization(path, txn) & mode == mode {
                self.hypothetical
                    .route(path)
                    .ok_or_else(|| TCError::not_found(TCPath::from(&path[2..])))
            } else {
                Err(unauthorized!("access to {}", TCPath::from(path)))
            }
        } else {
            Err(not_found!("{}", TCPath::from(path)))
        }
    }
}

impl<FE> Kernel<FE> {
    pub(crate) async fn commit(&self, txn_id: TxnId) {
        self.hypothetical.rollback(&txn_id).await;
    }

    pub(crate) async fn finalize(&self, txn_id: TxnId) {
        self.hypothetical.finalize(&txn_id).await;
    }
}

#[async_trait]
impl<FE: ThreadSafe + Clone> fs::Persist<FE> for Kernel<FE> {
    type Schema = (Option<Link>, Option<Link>);
    type Txn = Txn<FE>;

    async fn create(_txn_id: TxnId, schema: Self::Schema, _store: fs::Dir<FE>) -> TCResult<Self> {
        let (owner, group) = schema;
        let schema = Schema::new(Hypothetical::PATH, owner, group);

        Ok(Self {
            hypothetical: Cluster::new(schema, Hypothetical::new()),
            file: PhantomData,
        })
    }

    async fn load(_txn_id: TxnId, schema: Self::Schema, _store: fs::Dir<FE>) -> TCResult<Self> {
        let (owner, group) = schema;
        let schema = Schema::new(Hypothetical::PATH, owner, group);

        Ok(Self {
            hypothetical: Cluster::new(schema, Hypothetical::new()),
            file: PhantomData,
        })
    }

    fn dir(&self) -> fs::Inner<FE> {
        unimplemented!("Kernel::inner")
    }
}
