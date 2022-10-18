use futures::TryFutureExt;

use tc_error::*;
use tc_transact::TxnId;

use crate::route::{Public, Route};
use crate::txn::Txn;

pub use block::{ChainBlock, Mutation};
pub use history::{History, HistoryView};
pub(super) use store::Store;

mod block;
mod history;
mod store;

pub(super) async fn replay_all<T>(
    subject: &T,
    past_txn_id: &TxnId,
    mutations: &[Mutation],
    txn: &Txn,
    store: &Store,
) -> TCResult<()>
where
    T: Route + Public,
{
    for op in mutations {
        replay(subject, txn, &store, op)
            .map_err(|err| err.consume(format!("while replaying transaction {}", past_txn_id)))
            .await?;
    }

    Ok(())
}

async fn replay<T>(subject: &T, txn: &Txn, store: &Store, mutation: &Mutation) -> TCResult<()>
where
    T: Route + Public,
{
    match mutation {
        Mutation::Delete(path, key) => subject.delete(txn, path, key.clone()).await,
        Mutation::Put(path, key, value) => {
            let value = store.resolve(txn, value.clone()).await?;
            subject.put(txn, path, key.clone(), value.clone()).await
        }
    }
}
