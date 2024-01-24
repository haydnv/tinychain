pub use block::CacheBlock;
pub use txn::{Actor, Gateway, Resolver, SignedToken, Token, Txn, TxnServer};

pub use txn::hypothetical;

mod block;
mod txn;

/// A transactional directory
pub type Dir = tc_transact::fs::Dir<CacheBlock>;

/// An entry in a transactional directory
pub type DirEntry<B> = tc_transact::fs::DirEntry<CacheBlock, B>;

/// A transactional file
pub type File<B> = tc_transact::fs::File<CacheBlock, B>;
