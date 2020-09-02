use std::convert::TryInto;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use futures::TryFutureExt;

use crate::class::{Class, Instance, TCResult, TCStream};
use crate::collection::class::*;
use crate::collection::{Collection, CollectionBase, CollectionView};
use crate::error;
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::class::NumberType;
use crate::value::{label, Link, Number, TCPath, Value};

use super::bounds::{Bounds, Shape};
use super::dense::BlockListFile;
use super::sparse::SparseTable;
use super::{DenseTensor, SparseTensor, TensorBoolean, TensorIO, TensorTransform};

pub trait TensorInstance: Send + Sync {
    fn dtype(&self) -> NumberType;

    fn ndim(&self) -> usize;

    fn shape(&'_ self) -> &'_ Shape;

    fn size(&self) -> u64;
}

#[derive(Clone, Eq, PartialEq)]
pub enum TensorBaseType {
    Dense,
    Sparse,
}

impl Class for TensorBaseType {
    type Instance = TensorBase;

    fn from_path(path: &TCPath) -> TCResult<Self> {
        let suffix = path.from_path(&Self::prefix())?;
        if suffix.len() == 1 {
            match suffix[0].as_str() {
                "dense" => Ok(TensorBaseType::Dense),
                "sparse" => Ok(TensorBaseType::Sparse),
                other => Err(error::not_found(other)),
            }
        } else {
            Err(error::not_found(suffix))
        }
    }

    fn prefix() -> TCPath {
        CollectionBaseType::prefix().join(label("tensor").into())
    }
}

impl From<TensorBaseType> for Link {
    fn from(tbt: TensorBaseType) -> Link {
        let prefix = TensorBaseType::prefix();

        use TensorBaseType::*;
        match tbt {
            Dense => prefix.join(label("dense").into()).into(),
            Sparse => prefix.join(label("sparse").into()).into(),
        }
    }
}

impl fmt::Display for TensorBaseType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Dense => write!(f, "type: DenseTensor"),
            Self::Sparse => write!(f, "type: SparseTensor"),
        }
    }
}

#[derive(Clone)]
pub enum TensorBase {
    Dense(BlockListFile),
    Sparse(SparseTable),
}

impl Instance for TensorBase {
    type Class = TensorBaseType;

    fn class(&self) -> Self::Class {
        match self {
            Self::Dense(_) => Self::Class::Dense,
            Self::Sparse(_) => Self::Class::Sparse,
        }
    }
}

#[async_trait]
impl CollectionInstance for TensorBase {
    type Error = error::TCError;
    type Item = Number;
    type Slice = TensorView;

    async fn get(&self, txn: Arc<Txn>, selector: Value) -> TCResult<Self::Slice> {
        TensorView::from(self.clone()).get(txn, selector).await
    }

    async fn is_empty(&self, txn: Arc<Txn>) -> TCResult<bool> {
        TensorView::from(self.clone()).is_empty(txn).await
    }

    async fn put(&self, txn: Arc<Txn>, selector: Value, value: Self::Item) -> TCResult<()> {
        TensorView::from(self.clone())
            .put(txn, selector, value)
            .await
    }

    async fn to_stream(&self, txn: Arc<Txn>) -> TCResult<TCStream<Self::Item>> {
        TensorView::from(self.clone()).to_stream(txn).await
    }
}

impl TensorInstance for TensorBase {
    fn dtype(&self) -> NumberType {
        match self {
            Self::Dense(dense) => dense.dtype(),
            Self::Sparse(sparse) => sparse.dtype(),
        }
    }

    fn ndim(&self) -> usize {
        match self {
            Self::Dense(dense) => dense.ndim(),
            Self::Sparse(sparse) => sparse.ndim(),
        }
    }

    fn shape(&'_ self) -> &'_ Shape {
        match self {
            Self::Dense(dense) => dense.shape(),
            Self::Sparse(sparse) => sparse.shape(),
        }
    }

    fn size(&self) -> u64 {
        match self {
            Self::Dense(dense) => dense.size(),
            Self::Sparse(sparse) => sparse.size(),
        }
    }
}

#[async_trait]
impl Transact for TensorBase {
    async fn commit(&self, txn_id: &TxnId) {
        match self {
            Self::Dense(dense) => dense.commit(txn_id).await,
            Self::Sparse(sparse) => sparse.commit(txn_id).await,
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        match self {
            Self::Dense(dense) => dense.rollback(txn_id).await,
            Self::Sparse(sparse) => sparse.rollback(txn_id).await,
        }
    }
}

impl From<TensorBase> for Collection {
    fn from(base: TensorBase) -> Collection {
        Collection::Base(CollectionBase::Tensor(base))
    }
}

impl From<TensorBase> for TensorView {
    fn from(base: TensorBase) -> TensorView {
        match base {
            TensorBase::Dense(blocks) => Self::Dense(blocks.into()),
            TensorBase::Sparse(table) => Self::Sparse(table.into()),
        }
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum TensorViewType {
    Dense,
    Sparse,
}

impl Class for TensorViewType {
    type Instance = TensorView;

    fn from_path(path: &TCPath) -> TCResult<Self> {
        Err(error::bad_request(crate::class::ERR_PROTECTED, path))
    }

    fn prefix() -> TCPath {
        TensorBaseType::prefix()
    }
}

impl From<TensorViewType> for Link {
    fn from(tvt: TensorViewType) -> Link {
        let prefix = TensorViewType::prefix();

        use TensorViewType::*;
        match tvt {
            Dense => prefix.join(label("dense").into()).into(),
            Sparse => prefix.join(label("sparse").into()).into(),
        }
    }
}

impl fmt::Display for TensorViewType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Dense => write!(f, "type: DenseTensorView"),
            Self::Sparse => write!(f, "type: SparseTensorView"),
        }
    }
}

#[derive(Clone)]
pub enum TensorView {
    Dense(DenseTensor),
    Sparse(SparseTensor),
}

impl Instance for TensorView {
    type Class = TensorViewType;

    fn class(&self) -> Self::Class {
        match self {
            Self::Dense(_) => Self::Class::Dense,
            Self::Sparse(_) => Self::Class::Sparse,
        }
    }
}

#[async_trait]
impl CollectionInstance for TensorView {
    type Error = error::TCError;
    type Item = Number;
    type Slice = TensorView;

    async fn get(&self, _txn: Arc<Txn>, selector: Value) -> TCResult<Self::Slice> {
        let bounds: Bounds = selector.try_into()?;
        self.slice(bounds).map(TensorView::from)
    }

    async fn is_empty(&self, txn: Arc<Txn>) -> TCResult<bool> {
        self.any(txn).map_ok(|any| !any).await
    }

    async fn put(&self, txn: Arc<Txn>, selector: Value, value: Self::Item) -> TCResult<()> {
        let bounds: Bounds = selector.try_into()?;
        self.write_value(txn.id().clone(), bounds, value).await
    }

    async fn to_stream(&self, _txn: Arc<Txn>) -> TCResult<TCStream<Self::Item>> {
        Err(error::not_implemented("TensorView::to_stream"))
    }
}

impl TensorInstance for TensorView {
    fn dtype(&self) -> NumberType {
        match self {
            Self::Dense(dense) => dense.dtype(),
            Self::Sparse(sparse) => sparse.dtype(),
        }
    }

    fn ndim(&self) -> usize {
        match self {
            Self::Dense(dense) => dense.ndim(),
            Self::Sparse(sparse) => sparse.ndim(),
        }
    }

    fn shape(&'_ self) -> &'_ Shape {
        match self {
            Self::Dense(dense) => dense.shape(),
            Self::Sparse(sparse) => sparse.shape(),
        }
    }

    fn size(&self) -> u64 {
        match self {
            Self::Dense(dense) => dense.size(),
            Self::Sparse(sparse) => sparse.size(),
        }
    }
}

#[async_trait]
impl Transact for TensorView {
    async fn commit(&self, txn_id: &TxnId) {
        match self {
            Self::Dense(dense) => dense.commit(txn_id).await,
            Self::Sparse(sparse) => sparse.commit(txn_id).await,
        }
    }

    async fn rollback(&self, txn_id: &TxnId) {
        match self {
            Self::Dense(dense) => dense.rollback(txn_id).await,
            Self::Sparse(sparse) => sparse.rollback(txn_id).await,
        }
    }
}

impl From<DenseTensor> for TensorView {
    fn from(dense: DenseTensor) -> TensorView {
        Self::Dense(dense)
    }
}

impl From<SparseTensor> for TensorView {
    fn from(sparse: SparseTensor) -> TensorView {
        Self::Sparse(sparse)
    }
}

impl From<TensorView> for Collection {
    fn from(view: TensorView) -> Collection {
        Collection::View(CollectionView::Tensor(view))
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum TensorType {
    Base,
    View,
}

#[derive(Clone)]
pub enum Tensor {
    Base(TensorBase),
    View(TensorView),
}

impl TensorInstance for Tensor {
    fn dtype(&self) -> NumberType {
        match self {
            Self::Base(base) => base.dtype(),
            Self::View(view) => view.dtype(),
        }
    }

    fn ndim(&self) -> usize {
        match self {
            Self::Base(base) => base.ndim(),
            Self::View(view) => view.ndim(),
        }
    }

    fn shape(&'_ self) -> &'_ Shape {
        match self {
            Self::Base(base) => base.shape(),
            Self::View(view) => view.shape(),
        }
    }

    fn size(&self) -> u64 {
        match self {
            Self::Base(base) => base.size(),
            Self::View(view) => view.size(),
        }
    }
}

#[async_trait]
impl Transact for Tensor {
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
