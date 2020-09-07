use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::StreamExt;

use crate::class::{Instance, TCResult, TCStream};
use crate::error;
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::Value;

pub mod btree;
pub mod chain;
pub mod class;
pub mod cluster;
pub mod graph;
pub mod schema;
pub mod table;
pub mod tensor;

pub type CollectionType = class::CollectionType;
pub type CollectionBaseType = class::CollectionBaseType;
pub type CollectionItem<I, S> = class::CollectionItem<I, S>;
pub type CollectionViewType = class::CollectionViewType;

pub type BTree = btree::BTree;
pub type BTreeSlice = btree::BTreeSlice;
pub type Chain = chain::Chain;
pub type Cluster = cluster::Cluster;
pub type Graph = graph::Graph;
pub type Table = table::Table;
pub type Tensor = tensor::Tensor;
pub type TensorView = tensor::TensorView;

#[derive(Clone)]
pub enum CollectionBase {
    BTree(btree::BTreeFile),
    Chain(chain::Chain),
    Graph(graph::Graph),
    Table(table::TableBase),
    Tensor(tensor::class::TensorBase),
}

impl Instance for CollectionBase {
    type Class = class::CollectionBaseType;

    fn class(&self) -> Self::Class {
        match self {
            Self::BTree(_) => class::CollectionBaseType::BTree,
            Self::Chain(chain) => class::CollectionBaseType::Chain(chain.class()),
            Self::Graph(_) => class::CollectionBaseType::Graph,
            Self::Table(table) => class::CollectionBaseType::Table(table.class()),
            Self::Tensor(tensor) => class::CollectionBaseType::Tensor(tensor.class()),
        }
    }
}

#[async_trait]
impl class::CollectionInstance for CollectionBase {
    type Item = Value;
    type Slice = CollectionView;

    async fn get(
        &self,
        txn: Arc<Txn>,
        selector: Value,
    ) -> TCResult<CollectionItem<Self::Item, Self::Slice>> {
        let view: CollectionView = self.clone().into();
        view.get(txn, selector).await
    }

    async fn is_empty(&self, txn: Arc<Txn>) -> TCResult<bool> {
        match self {
            Self::BTree(btree) => btree.is_empty(txn).await,
            _ => Err(error::not_implemented("CollectionBase::is_empty")),
        }
    }

    async fn put(
        &self,
        txn: Arc<Txn>,
        selector: Value,
        value: CollectionItem<Self::Item, Self::Slice>,
    ) -> TCResult<()> {
        let view: CollectionView = self.clone().into();
        view.put(txn, selector, value).await
    }

    async fn to_stream(&self, txn: Arc<Txn>) -> TCResult<TCStream<Value>> {
        match self {
            Self::BTree(btree) => btree.to_stream(txn).await,
            Self::Chain(chain) => chain.to_stream(txn).await,
            Self::Graph(graph) => graph.to_stream(txn).await,
            Self::Table(table) => table.to_stream(txn).await,
            Self::Tensor(tensor) => tensor.to_stream(txn).await,
        }
    }
}

#[async_trait]
impl Transact for CollectionBase {
    async fn commit(&self, txn_id: &TxnId) {
        match self {
            Self::BTree(btree) => btree.commit(txn_id).await,
            Self::Chain(chain) => chain.commit(txn_id).await,
            Self::Graph(graph) => graph.commit(txn_id).await,
            Self::Table(table) => table.commit(txn_id).await,
            Self::Tensor(tensor) => tensor.commit(txn_id).await,
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        match self {
            Self::BTree(btree) => btree.rollback(txn_id).await,
            Self::Chain(chain) => chain.rollback(txn_id).await,
            Self::Graph(graph) => graph.rollback(txn_id).await,
            Self::Table(table) => table.rollback(txn_id).await,
            Self::Tensor(tensor) => tensor.rollback(txn_id).await,
        }
    }
}

impl From<CollectionBase> for CollectionView {
    fn from(base: CollectionBase) -> CollectionView {
        match base {
            CollectionBase::BTree(btree) => CollectionView::BTree(btree.into()),
            CollectionBase::Chain(chain) => CollectionView::Chain(chain),
            CollectionBase::Graph(graph) => CollectionView::Graph(graph),
            CollectionBase::Table(table) => CollectionView::Table(table.into()),
            CollectionBase::Tensor(tensor) => CollectionView::Tensor(tensor.into()),
        }
    }
}

impl fmt::Display for CollectionBase {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BTree(_) => write!(f, "(B-tree)"),
            Self::Chain(_) => write!(f, "(chain)"),
            Self::Graph(_) => write!(f, "(graph)"),
            Self::Table(_) => write!(f, "(table)"),
            Self::Tensor(_) => write!(f, "(tensor)"),
        }
    }
}

#[derive(Clone)]
pub enum CollectionView {
    BTree(btree::BTree),
    Chain(chain::Chain),
    Graph(graph::Graph),
    Table(table::Table),
    Tensor(tensor::Tensor),
}

impl Instance for CollectionView {
    type Class = class::CollectionViewType;

    fn class(&self) -> Self::Class {
        match self {
            Self::BTree(btree) => btree.class().into(),
            Self::Chain(chain) => chain.class().into(),
            Self::Graph(graph) => graph.class().into(),
            Self::Table(table) => table.class().into(),
            Self::Tensor(tensor) => tensor.class().into(),
        }
    }
}

#[async_trait]
impl class::CollectionInstance for CollectionView {
    type Item = Value;
    type Slice = CollectionView;

    async fn get(
        &self,
        txn: Arc<Txn>,
        selector: Value,
    ) -> TCResult<CollectionItem<Self::Item, Self::Slice>> {
        match self {
            Self::BTree(btree) => {
                let item = match btree.get(txn, selector).await? {
                    CollectionItem::Value(key) => CollectionItem::Value(Value::Tuple(key)),
                    CollectionItem::Slice(slice) => CollectionItem::Slice(slice.into()),
                };
                Ok(item)
            }
            _ => Err(error::not_implemented("CollectionView::get")),
        }
    }

    async fn is_empty(&self, txn: Arc<Txn>) -> TCResult<bool> {
        match self {
            Self::BTree(btree) => btree.is_empty(txn).await,
            _ => Err(error::not_implemented("CollectionView::is_empty")),
        }
    }

    async fn put(
        &self,
        txn: Arc<Txn>,
        selector: Value,
        value: CollectionItem<Self::Item, Self::Slice>,
    ) -> TCResult<()> {
        match self {
            Self::BTree(btree) => match value {
                CollectionItem::Value(value) => {
                    let value = value.try_into()?;
                    btree.put(txn, selector, CollectionItem::Value(value)).await
                }
                CollectionItem::Slice(slice) => {
                    let slice = slice.try_into()?;
                    btree.put(txn, selector, CollectionItem::Slice(slice)).await
                }
            },
            Self::Table(table) => match value {
                CollectionItem::Value(value) => {
                    let value = value.try_into()?;
                    table.put(txn, selector, CollectionItem::Value(value)).await
                }
                CollectionItem::Slice(slice) => {
                    let slice = slice.try_into()?;
                    table.put(txn, selector, CollectionItem::Slice(slice)).await
                }
            },
            Self::Tensor(tensor) => match value {
                CollectionItem::Value(value) => {
                    let value = value.try_into()?;
                    tensor
                        .put(txn, selector, CollectionItem::Value(value))
                        .await
                }
                CollectionItem::Slice(slice) => {
                    let slice = slice.try_into()?;
                    tensor
                        .put(txn, selector, CollectionItem::Slice(slice))
                        .await
                }
            },
            _ => Err(error::not_implemented("CollectionView::put")),
        }
    }

    async fn to_stream(&self, txn: Arc<Txn>) -> TCResult<TCStream<Self::Item>> {
        match self {
            Self::BTree(btree) => {
                let stream = btree.to_stream(txn).await?;
                Ok(Box::pin(stream.map(Value::from)))
            }
            _ => Err(error::not_implemented("CollectionView::stream")),
        }
    }
}

#[async_trait]
impl Transact for CollectionView {
    async fn commit(&self, txn_id: &TxnId) {
        match self {
            Self::BTree(btree) => btree.commit(txn_id).await,
            Self::Chain(chain) => chain.commit(txn_id).await,
            Self::Graph(graph) => graph.commit(txn_id).await,
            Self::Table(table) => table.commit(txn_id).await,
            Self::Tensor(tensor) => tensor.commit(txn_id).await,
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        match self {
            Self::BTree(btree) => btree.rollback(txn_id).await,
            Self::Chain(chain) => chain.rollback(txn_id).await,
            Self::Graph(graph) => graph.rollback(txn_id).await,
            Self::Table(table) => table.rollback(txn_id).await,
            Self::Tensor(tensor) => tensor.rollback(txn_id).await,
        }
    }
}

impl TryFrom<CollectionView> for BTree {
    type Error = error::TCError;

    fn try_from(view: CollectionView) -> TCResult<BTree> {
        match view {
            CollectionView::BTree(btree) => Ok(btree),
            other => Err(error::bad_request("Expected BTree but found", other)),
        }
    }
}

impl TryFrom<CollectionView> for BTreeSlice {
    type Error = error::TCError;

    fn try_from(view: CollectionView) -> TCResult<BTreeSlice> {
        match view {
            CollectionView::BTree(btree) => Ok(btree.into()),
            other => Err(error::bad_request("Expected BTree but found", other)),
        }
    }
}

impl fmt::Display for CollectionView {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BTree(_) => write!(f, "(B-tree view)"),
            Self::Chain(_) => write!(f, "(chain)"),
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
    type Item = Value;
    type Slice = CollectionView;

    async fn get(
        &self,
        txn: Arc<Txn>,
        selector: Value,
    ) -> TCResult<CollectionItem<Self::Item, Self::Slice>> {
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

    async fn put(
        &self,
        txn: Arc<Txn>,
        selector: Value,
        value: CollectionItem<Self::Item, Self::Slice>,
    ) -> TCResult<()> {
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
