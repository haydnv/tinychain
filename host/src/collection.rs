use crate::fs::CacheBlock;
use crate::txn::Txn;

pub use tc_collection::{CollectionType, Schema};

/// A collection such as a table or tensor
pub type Collection = tc_collection::Collection<Txn, CacheBlock>;

/// The base type of a [`Collection`]
pub type CollectionBase = tc_collection::CollectionBase<Txn, CacheBlock>;

/// A view of a [`Collection`] which supports stream-encoding
pub type CollectionView<'en> = tc_collection::CollectionView<'en, Txn, CacheBlock>;
