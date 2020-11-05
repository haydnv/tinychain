use std::convert::{TryFrom, TryInto};
use std::fmt;

use async_trait::async_trait;

use crate::class::{Instance, TCResult, TCStream};
use crate::error;
use crate::scalar::{Number, Scalar, TCPath, Value};
use crate::transaction::{Transact, Txn, TxnId};

pub mod btree;
pub mod class;
pub mod null;
pub mod schema;
pub mod table;
pub mod tensor;

pub type CollectionType = class::CollectionType;
pub type CollectionBaseType = class::CollectionBaseType;
pub type CollectionItem<I, S> = class::CollectionItem<I, S>;

pub type BTree = btree::BTree;
pub type BTreeSlice = btree::BTreeSlice;

#[derive(Clone)]
pub enum CollectionBase {
    BTree(btree::BTreeFile),
    Null(null::Null),
    Table(table::TableBase),
    Tensor(tensor::class::TensorBase),
}

impl Instance for CollectionBase {
    type Class = class::CollectionBaseType;

    fn class(&self) -> Self::Class {
        match self {
            Self::BTree(_) => class::CollectionBaseType::BTree,
            Self::Null(_) => class::CollectionBaseType::Null,
            Self::Table(table) => class::CollectionBaseType::Table(table.class()),
            Self::Tensor(tensor) => class::CollectionBaseType::Tensor(tensor.class()),
        }
    }
}

#[async_trait]
impl class::CollectionInstance for CollectionBase {
    type Item = Scalar;
    type Slice = CollectionView;

    async fn get(
        &self,
        txn: Txn,
        path: TCPath,
        selector: Value,
    ) -> TCResult<CollectionItem<Self::Item, Self::Slice>> {
        let view: CollectionView = self.clone().into();
        view.get(txn, path, selector).await
    }

    async fn is_empty(&self, txn: &Txn) -> TCResult<bool> {
        match self {
            Self::BTree(btree) => btree.is_empty(txn).await,
            Self::Null(null) => null.is_empty(txn).await,
            Self::Table(table) => table.is_empty(txn).await,
            Self::Tensor(tensor) => tensor.is_empty(txn).await,
        }
    }

    async fn put(
        &self,
        txn: Txn,
        path: TCPath,
        selector: Value,
        value: CollectionItem<Self::Item, Self::Slice>,
    ) -> TCResult<()> {
        let view: CollectionView = self.clone().into();
        view.put(txn, path, selector, value).await
    }

    async fn to_stream(&self, txn: Txn) -> TCResult<TCStream<Scalar>> {
        match self {
            Self::BTree(btree) => btree.to_stream(txn).await,
            Self::Null(null) => null.to_stream(txn).await,
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
            Self::Null(null) => null.commit(txn_id).await,
            Self::Table(table) => table.commit(txn_id).await,
            Self::Tensor(tensor) => tensor.commit(txn_id).await,
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        match self {
            Self::BTree(btree) => btree.rollback(txn_id).await,
            Self::Null(null) => null.rollback(txn_id).await,
            Self::Table(table) => table.rollback(txn_id).await,
            Self::Tensor(tensor) => tensor.rollback(txn_id).await,
        }
    }

    async fn finalize(&self, txn_id: &TxnId) {
        match self {
            Self::BTree(btree) => btree.finalize(txn_id).await,
            Self::Null(null) => null.finalize(txn_id).await,
            Self::Table(table) => table.finalize(txn_id).await,
            Self::Tensor(tensor) => tensor.finalize(txn_id).await,
        }
    }
}

impl From<CollectionBase> for CollectionView {
    fn from(base: CollectionBase) -> CollectionView {
        match base {
            CollectionBase::BTree(btree) => CollectionView::BTree(btree.into()),
            CollectionBase::Null(null) => CollectionView::Null(null),
            CollectionBase::Table(table) => CollectionView::Table(table.into()),
            CollectionBase::Tensor(tensor) => CollectionView::Tensor(tensor.into()),
        }
    }
}

impl fmt::Display for CollectionBase {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::BTree(_) => write!(f, "(B-tree)"),
            Self::Null(_) => write!(f, "(null)"),
            Self::Table(_) => write!(f, "(table)"),
            Self::Tensor(_) => write!(f, "(tensor)"),
        }
    }
}

#[derive(Clone)]
pub enum CollectionView {
    BTree(btree::BTree),
    Null(null::Null),
    Table(table::Table),
    Tensor(tensor::Tensor),
}

impl Instance for CollectionView {
    type Class = class::CollectionViewType;

    fn class(&self) -> Self::Class {
        match self {
            Self::BTree(btree) => btree.class().into(),
            Self::Null(null) => null.class().into(),
            Self::Table(table) => table.class().into(),
            Self::Tensor(tensor) => tensor.class().into(),
        }
    }
}

#[async_trait]
impl class::CollectionInstance for CollectionView {
    type Item = Scalar;
    type Slice = CollectionView;

    async fn get(
        &self,
        txn: Txn,
        path: TCPath,
        selector: Value,
    ) -> TCResult<CollectionItem<Self::Item, Self::Slice>> {
        match self {
            Self::BTree(btree) => {
                let item = match btree.get(txn, path, selector).await? {
                    CollectionItem::Scalar(key) => {
                        CollectionItem::Scalar(Scalar::Value(Value::Tuple(key)))
                    }
                    CollectionItem::Slice(slice) => CollectionItem::Slice(slice.into()),
                };

                Ok(item)
            }
            _ => Err(error::not_implemented("CollectionView::get")),
        }
    }

    async fn is_empty(&self, txn: &Txn) -> TCResult<bool> {
        match self {
            Self::BTree(btree) => btree.is_empty(txn).await,
            Self::Null(null) => null.is_empty(txn).await,
            Self::Table(table) => table.is_empty(txn).await,
            Self::Tensor(tensor) => tensor.is_empty(txn).await,
        }
    }

    async fn put(
        &self,
        txn: Txn,
        path: TCPath,
        selector: Value,
        value: CollectionItem<Self::Item, Self::Slice>,
    ) -> TCResult<()> {
        match self {
            Self::BTree(btree) => match value {
                CollectionItem::Scalar(value) => {
                    let value = value.try_into()?;
                    btree
                        .put(txn, path, selector, CollectionItem::Scalar(value))
                        .await
                }
                CollectionItem::Slice(slice) => {
                    let slice = slice.try_into()?;
                    btree
                        .put(txn, path, selector, CollectionItem::Slice(slice))
                        .await
                }
            },
            Self::Null(_) => Err(error::unsupported("Cannot modify a Null Collection")),
            Self::Table(table) => match value {
                CollectionItem::Scalar(value) => {
                    let value = value.try_into()?;
                    table
                        .put(txn, path, selector, CollectionItem::Scalar(value))
                        .await
                }
                CollectionItem::Slice(slice) => {
                    let slice = slice.try_into()?;
                    table
                        .put(txn, path, selector, CollectionItem::Slice(slice))
                        .await
                }
            },
            Self::Tensor(tensor) => match value {
                CollectionItem::Scalar(value) => {
                    let value = Value::try_from(value)?;
                    let number = Number::try_from(value)?;
                    tensor
                        .put(txn, path, selector, CollectionItem::Scalar(number))
                        .await
                }
                CollectionItem::Slice(slice) => {
                    let slice = slice.try_into()?;
                    tensor
                        .put(txn, path, selector, CollectionItem::Slice(slice))
                        .await
                }
            },
        }
    }

    async fn to_stream(&self, txn: Txn) -> TCResult<TCStream<Scalar>> {
        match self {
            Self::BTree(btree) => btree.to_stream(txn).await,
            Self::Null(null) => null.to_stream(txn).await,
            Self::Table(table) => table.to_stream(txn).await,
            Self::Tensor(tensor) => tensor.to_stream(txn).await,
        }
    }
}

#[async_trait]
impl Transact for CollectionView {
    async fn commit(&self, txn_id: &TxnId) {
        match self {
            Self::BTree(btree) => btree.commit(txn_id).await,
            Self::Null(null) => null.commit(txn_id).await,
            Self::Table(table) => table.commit(txn_id).await,
            Self::Tensor(tensor) => tensor.commit(txn_id).await,
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        match self {
            Self::BTree(btree) => btree.rollback(txn_id).await,
            Self::Null(null) => null.rollback(txn_id).await,
            Self::Table(table) => table.rollback(txn_id).await,
            Self::Tensor(tensor) => tensor.rollback(txn_id).await,
        }
    }

    async fn finalize(&self, txn_id: &TxnId) {
        match self {
            Self::BTree(btree) => btree.finalize(txn_id).await,
            Self::Null(null) => null.finalize(txn_id).await,
            Self::Table(table) => table.finalize(txn_id).await,
            Self::Tensor(tensor) => tensor.finalize(txn_id).await,
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
            Self::Null(_) => write!(f, "(null collection)"),
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
    type Item = Scalar;
    type Slice = CollectionView;

    async fn get(
        &self,
        txn: Txn,
        path: TCPath,
        selector: Value,
    ) -> TCResult<CollectionItem<Self::Item, Self::Slice>> {
        match self {
            Self::Base(base) => base.get(txn, path, selector).await,
            Self::View(view) => view.get(txn, path, selector).await,
        }
    }

    async fn is_empty(&self, txn: &Txn) -> TCResult<bool> {
        match self {
            Self::Base(base) => base.is_empty(txn).await,
            Self::View(view) => view.is_empty(txn).await,
        }
    }

    async fn put(
        &self,
        txn: Txn,
        path: TCPath,
        selector: Value,
        value: CollectionItem<Self::Item, Self::Slice>,
    ) -> TCResult<()> {
        match self {
            Self::Base(base) => base.put(txn, path, selector, value).await,
            Self::View(view) => view.put(txn, path, selector, value).await,
        }
    }

    async fn to_stream(&self, txn: Txn) -> TCResult<TCStream<Scalar>> {
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

    async fn finalize(&self, txn_id: &TxnId) {
        match self {
            Self::Base(base) => base.finalize(txn_id).await,
            Self::View(view) => view.finalize(txn_id).await,
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
