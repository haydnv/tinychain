use tc_fs::CacheBlock;

use crate::txn::Txn;

/// A collection such as a [`Table`] or [`Tensor`]
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

/// An n-dimensional array
pub type Tensor = tc_collection::tensor::Tensor<Txn, CacheBlock>;

/// A tensor file
pub type TensorBase = tc_collection::tensor::TensorBase<Txn, CacheBlock>;

/// A dense n-dimensional array
pub type Dense = tc_collection::tensor::Dense<Txn, CacheBlock>;

/// A dense tensor file
pub type DenseBase = tc_collection::tensor::DenseBase<Txn, CacheBlock>;

/// A sparse n-dimensional array
pub type Sparse = tc_collection::tensor::Sparse<Txn, CacheBlock>;

/// A sparse tensor file
pub type SparseBase = tc_collection::tensor::SparseBase<Txn, CacheBlock>;
