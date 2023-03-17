use crate::fs::CacheBlock;
use crate::txn::Txn;

/// A collection such as a table or tensor
pub type Collection = tc_collection::Collection<Txn, CacheBlock>;

/// The base type of a [`Collection`]
pub type CollectionBase = tc_collection::CollectionBase<Txn, CacheBlock>;

/// A view of a [`Collection`] which supports stream-encoding
pub type CollectionView<'en> = tc_collection::CollectionView<'en>;

/// A B+Tree
pub type BTree = tc_collection::BTree<Txn, CacheBlock>;

/// A B+Tree file
pub type BTreeFile = tc_collection::btree::BTreeFile<Txn, CacheBlock>;

/// A relational database table
pub type Table = tc_collection::table::Table<Txn, CacheBlock>;

/// A relational database table file
pub type TableFile = tc_collection::table::TableFile<Txn, CacheBlock>;
