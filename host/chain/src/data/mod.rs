use std::fmt;

use futures::TryFutureExt;

use tc_collection::Collection;
use tc_error::*;
use tc_scalar::Scalar;
use tc_transact::public::{Public, Route, StateInstance};
use tc_transact::{Transaction, TxnId};

pub use block::{ChainBlock, MutationRecord};
pub use history::{History, HistoryView};
pub(super) use store::{Store, StoreEntry};

mod block;
mod history;
mod store;

pub(super) async fn replay_all<State, T>(
    subject: &T,
    past_txn_id: &TxnId,
    mutations: &[MutationRecord],
    txn: &State::Txn,
    store: &Store<State::Txn, State::FE>,
) -> TCResult<()>
where
    State: StateInstance + From<Collection<State::Txn, State::FE>> + From<Scalar>,
    T: Route<State> + fmt::Debug,
{
    for op in mutations {
        replay(subject, txn, &store, op)
            .map_err(|err| err.consume(format!("while replaying transaction {}", past_txn_id)))
            .await?;
    }

    Ok(())
}

async fn replay<State, T>(
    subject: &T,
    txn: &State::Txn,
    store: &Store<State::Txn, State::FE>,
    mutation: &MutationRecord,
) -> TCResult<()>
where
    State: StateInstance + From<Collection<State::Txn, State::FE>> + From<Scalar>,
    T: Route<State> + fmt::Debug,
{
    match mutation {
        MutationRecord::Delete(key) => subject.delete(txn, &[], key.clone()).await,
        MutationRecord::Put(key, value) => {
            let value = store.resolve(*txn.id(), value.clone()).await?;
            subject
                .put(txn, &[], key.clone(), value.into_state::<State>())
                .await
        }
    }
}
