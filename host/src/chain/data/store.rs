use std::fmt;
use std::iter::FromIterator;

use async_trait::async_trait;
use futures::future::TryFutureExt;
use log::debug;
use safecast::*;

use tc_collection::CollectionType;
use tc_error::*;
use tc_transact::fs::*;
use tc_transact::{AsyncHash, Transact, Transaction};
use tc_value::Value;
use tcgeneric::{Id, NativeClass};

use crate::collection::Collection;
use crate::fs;
use crate::scalar::{OpRef, Scalar, TCRef};
use crate::state::State;
use crate::txn::{Txn, TxnId};

#[derive(Clone)]
pub struct Store {
    dir: fs::Dir,
}

impl Store {
    pub fn new(dir: fs::Dir) -> Self {
        Self { dir }
    }

    pub async fn save_state(&self, txn: &Txn, state: State) -> TCResult<Scalar> {
        debug!("chain data store saving state {:?}...", state);

        let hash = state.clone().hash(txn).map_ok(Id::from).await?;

        match state {
            State::Collection(_collection) => Err(not_implemented!("save collection state")),
            State::Scalar(value) => Ok(value),
            other => other.try_cast_into(|s| bad_request!("Chain does not support value {:?}", s)),
        }
    }

    pub async fn resolve(&self, txn_id: TxnId, scalar: Scalar) -> TCResult<State> {
        debug!("History::resolve {:?}", scalar);

        type OpSubject = crate::scalar::Subject;

        if let Scalar::Ref(tc_ref) = scalar {
            if let TCRef::Op(OpRef::Get((OpSubject::Ref(hash, classpath), schema))) = *tc_ref {
                let class = CollectionType::from_path(&classpath)
                    .ok_or_else(|| unexpected!("invalid Collection type: {}", classpath))?;

                Err(unimplemented!("resolve saved collection"))
            } else {
                Err(unexpected!(
                    "invalid subject for historical Chain state {:?}",
                    tc_ref
                ))
            }
        } else {
            Ok(scalar.into())
        }
    }
}

#[async_trait]
impl Transact for Store {
    type Commit = ();

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        debug!("commit chain data store at {}", txn_id);

        self.dir.commit(txn_id, true).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.dir.rollback(txn_id, true).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.dir.finalize(txn_id).await
    }
}
