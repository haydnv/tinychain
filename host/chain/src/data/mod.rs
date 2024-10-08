use std::fmt;

use freqfs::FileSave;
use futures::TryFutureExt;
use safecast::TryCastFrom;

use tc_collection::{Collection, CollectionBlock};
use tc_error::*;
use tc_scalar::Scalar;
use tc_transact::public::{Public, Route, StateInstance};
use tc_transact::{Transaction, TxnId};

pub use block::{ChainBlock, MutationPending, MutationRecord};
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
    State::FE: for<'a> FileSave<'a> + CollectionBlock + Clone,
    T: Route<State> + fmt::Debug,
    Collection<State::Txn, State::FE>: TryCastFrom<State>,
    Scalar: TryCastFrom<State>,
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
    State::FE: for<'a> FileSave<'a> + CollectionBlock + Clone,
    T: Route<State> + fmt::Debug,
    Collection<State::Txn, State::FE>: TryCastFrom<State>,
    Scalar: TryCastFrom<State>,
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
