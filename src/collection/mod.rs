use std::convert::{Infallible, TryInto};
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::StreamExt;

use crate::class::{Instance, TCResult, TCStream};
use crate::error;
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::Value;

pub mod btree;
pub mod class;
pub mod graph;
pub mod schema;
pub mod table;
pub mod tensor;

pub type CollectionType = class::CollectionType;

pub type BTree = btree::BTree;
pub type BTreeSlice = btree::BTreeSlice;
pub type Graph = graph::Graph;
pub type Table = table::Table;
pub type TensorView = tensor::TensorView;

#[derive(Clone)]
pub enum CollectionBase {
    BTree(btree::BTreeFile),
    Graph(graph::Graph),
    Table(table::TableBase),
    Tensor(tensor::TensorBase),
}

impl Instance for CollectionBase {
    type Class = class::CollectionBaseType;

    fn class(&self) -> Self::Class {
        match self {
            Self::BTree(_) => class::CollectionBaseType::BTree,
            Self::Graph(_) => class::CollectionBaseType::Graph, // TODO
            Self::Table(tbt) => class::CollectionBaseType::Table(tbt.class()),
            Self::Tensor(_) => class::CollectionBaseType::Tensor, // TODO
        }
    }
}

#[async_trait]
impl class::CollectionInstance for CollectionBase {
    type Error = Infallible;
    type Item = Value;
    type Slice = CollectionView;

    async fn get(&self, txn: Arc<Txn>, selector: Value) -> TCResult<Self::Slice> {
        match self {
            Self::BTree(btree) => btree
                .get(txn, selector)
                .await
                .map(BTree::Slice)
                .map(CollectionView::BTree),
            _ => Err(error::not_implemented("CollectionBase::get")),
        }
    }

    async fn is_empty(&self, txn: Arc<Txn>) -> TCResult<bool> {
        match self {
            Self::BTree(btree) => btree.is_empty(txn).await,
            _ => Err(error::not_implemented("CollectionBase::is_empty")),
        }
    }

    async fn put(&self, txn: Arc<Txn>, selector: Value, value: Self::Item) -> TCResult<()> {
        match self {
            Self::BTree(btree) => btree.put(txn, selector, value.try_into()?).await,
            _ => Err(error::not_implemented("CollectionBase::put")),
        }
    }

    async fn to_stream(&self, txn: Arc<Txn>) -> TCResult<TCStream<Self::Item>> {
        match self {
            Self::BTree(btree) => {
                let stream = btree.to_stream(txn).await?;
                Ok(Box::pin(stream.map(Value::from)))
            }
            _ => Err(error::not_implemented("CollectionBase::stream")),
        }
    }
}

#[async_trait]
impl Transact for CollectionBase {
    async fn commit(&self, txn_id: &TxnId) {
        match self {
            Self::BTree(btree) => btree.commit(txn_id).await,
            Self::Graph(graph) => graph.commit(txn_id).await,
            Self::Table(table) => table.commit(txn_id).await,
            Self::Tensor(tensor) => tensor.commit(txn_id).await,
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        match self {
            Self::BTree(btree) => btree.rollback(txn_id).await,
            Self::Graph(graph) => graph.rollback(txn_id).await,
            Self::Table(table) => table.rollback(txn_id).await,
            Self::Tensor(tensor) => tensor.rollback(txn_id).await,
        }
    }
}

impl fmt::Display for CollectionBase {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BTree(_) => write!(f, "(B-tree)"),
            Self::Graph(_) => write!(f, "(graph)"),
            Self::Table(_) => write!(f, "(table)"),
            Self::Tensor(_) => write!(f, "(tensor)"),
        }
    }
}

#[derive(Clone)]
pub enum CollectionView {
    BTree(btree::BTree),
    Graph(graph::Graph),
    Table(table::Table),
    Tensor(tensor::TensorView),
}

impl Instance for CollectionView {
    type Class = class::CollectionViewType;

    fn class(&self) -> Self::Class {
        match self {
            Self::BTree(btree) => btree.class().into(),
            _ => unimplemented!(), // TODO
        }
    }
}

#[async_trait]
impl class::CollectionInstance for CollectionView {
    type Error = Infallible;
    type Item = Value;
    type Slice = CollectionView;

    async fn get(&self, txn: Arc<Txn>, selector: Value) -> TCResult<Self::Slice> {
        match self {
            Self::BTree(btree) => btree
                .get(txn, selector.try_into()?)
                .await
                .map(BTree::Slice)
                .map(CollectionView::BTree),
            _ => Err(error::not_implemented("CollectionView::get")),
        }
    }

    async fn is_empty(&self, txn: Arc<Txn>) -> TCResult<bool> {
        match self {
            Self::BTree(btree) => btree.is_empty(txn).await,
            _ => Err(error::not_implemented("CollectionView::is_empty")),
        }
    }

    async fn put(&self, txn: Arc<Txn>, selector: Value, value: Self::Item) -> TCResult<()> {
        match self {
            Self::BTree(btree) => {
                btree
                    .put(txn, selector.try_into()?, value.try_into()?)
                    .await
            }
            _ => Err(error::not_implemented("CollectionView::put")),
        }
    }

    async fn to_stream(&self, txn: Arc<Txn>) -> TCResult<TCStream<Self::Item>> {
        match self {
            Self::BTree(btree) => {
                let stream = btree.to_stream(txn).await?;
                Ok(Box::pin(stream.map(Value::from)))
            }
            _ => Err(error::not_implemented("CollectionVieW::stream")),
        }
    }
}

#[async_trait]
impl Transact for CollectionView {
    async fn commit(&self, txn_id: &TxnId) {
        match self {
            Self::BTree(btree) => btree.commit(txn_id).await,
            Self::Graph(graph) => graph.commit(txn_id).await,
            Self::Table(table) => table.commit(txn_id).await,
            Self::Tensor(tensor) => tensor.commit(txn_id).await,
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        match self {
            Self::BTree(btree) => btree.rollback(txn_id).await,
            Self::Graph(graph) => graph.rollback(txn_id).await,
            Self::Table(table) => table.rollback(txn_id).await,
            Self::Tensor(tensor) => tensor.rollback(txn_id).await,
        }
    }
}

impl fmt::Display for CollectionView {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BTree(_) => write!(f, "(B-tree view)"),
            Self::Graph(_) => write!(f, "(graph)"),
            Self::Table(_) => write!(f, "(table view)"),
            Self::Tensor(_) => write!(f, "(tensor view)"),
        }
    }
}

#[derive(Clone)]
pub enum Collection {
    Base(CollectionBase),
    View(CollectionView),
}

impl Instance for Collection {
    type Class = CollectionType;

    fn class(&self) -> CollectionType {
        match self {
            Self::Base(base) => base.class().into(),
            Self::View(view) => view.class().into(),
        }
    }
}

#[async_trait]
impl class::CollectionInstance for Collection {
    type Error = Infallible;
    type Item = Value;
    type Slice = CollectionView;

    async fn get(&self, txn: Arc<Txn>, selector: Value) -> TCResult<Self::Slice> {
        match self {
            Self::Base(base) => base.get(txn, selector).await,
            Self::View(view) => view.get(txn, selector).await,
        }
    }

    async fn is_empty(&self, txn: Arc<Txn>) -> TCResult<bool> {
        match self {
            Self::Base(base) => base.is_empty(txn).await,
            Self::View(view) => view.is_empty(txn).await,
        }
    }

    async fn put(&self, txn: Arc<Txn>, selector: Value, value: Self::Item) -> TCResult<()> {
        match self {
            Self::Base(base) => base.put(txn, selector, value).await,
            Self::View(view) => view.put(txn, selector, value).await,
        }
    }

    async fn to_stream(&self, txn: Arc<Txn>) -> TCResult<TCStream<Self::Item>> {
        match self {
            Self::Base(base) => base.to_stream(txn).await,
            Self::View(view) => view.to_stream(txn).await,
        }
    }
}

#[async_trait]
impl Transact for Collection {
    async fn commit(&self, txn_id: &TxnId) {
        match self {
            Self::Base(base) => base.commit(txn_id).await,
            Self::View(view) => view.commit(txn_id).await,
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        match self {
            Self::Base(base) => base.rollback(txn_id).await,
            Self::View(view) => view.rollback(txn_id).await,
        }
    }
}

impl From<CollectionBase> for Collection {
    fn from(base: CollectionBase) -> Collection {
        Collection::Base(base)
    }
}

impl From<CollectionView> for Collection {
    fn from(view: CollectionView) -> Collection {
        Collection::View(view)
    }
}

impl fmt::Display for Collection {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Base(base) => write!(f, "{}", base),
            Self::View(view) => write!(f, "{}", view),
        }
    }
}
