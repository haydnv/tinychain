use crate::class::TCBoxFuture;

pub mod lock;
mod txn;

pub type Txn = txn::Txn;
pub type TxnId = txn::TxnId;

pub trait Transact: Send + Sync {
    fn commit<'a>(&'a self, txn_id: &'a TxnId) -> TCBoxFuture<'a, ()>;

    fn rollback<'a>(&'a self, txn_id: &'a TxnId) -> TCBoxFuture<'a, ()>;
}
