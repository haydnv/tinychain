use std::convert::TryInto;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use futures::TryFutureExt;

use crate::class::{Class, Instance, TCResult, TCStream};
use crate::collection::class::*;
use crate::collection::{
    Collection, CollectionBase, CollectionBaseType, CollectionType, CollectionView,
};
use crate::error;
use crate::transaction::{Transact, Txn, TxnId};
use crate::value::number::class::{NumberClass, NumberType};
use crate::value::{label, Link, Number, TCPath, Value};

use super::bounds::{Bounds, Shape};
use super::dense::BlockListFile;
use super::sparse::SparseTable;
use super::{DenseTensor, SparseTensor, TensorBoolean, TensorIO, TensorTransform};

const ERR_SPECIFY_TYPE: &str = "You must specify a type of tensor (tensor/dense or tensor/sparse)";

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

#[async_trait]
impl CollectionClass for TensorBaseType {
    type Instance = TensorBase;

    async fn get(txn: Arc<Txn>, path: &TCPath, schema: Value) -> TCResult<TensorBase> {
        if path.is_empty() {
            return Err(error::unsupported(ERR_SPECIFY_TYPE));
        }

        match path[0].as_str() {
            "dense" if path.len() == 1 => {
                let (dtype, shape): (NumberType, Shape) = schema.try_into()?;
                let block_list = BlockListFile::constant(txn, shape, dtype.zero()).await?;
                Ok(TensorBase::Dense(block_list))
            }
            "sparse" if path.len() == 1 => todo!(),
            other => Err(error::not_found(other)),
        }
    }
}

impl From<TensorBaseType> for CollectionType {
    fn from(tbt: TensorBaseType) -> CollectionType {
        CollectionType::Base(CollectionBaseType::Tensor(tbt))
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
    type Item = Number;
    type Slice = TensorView;

    async fn get(
        &self,
        txn: Arc<Txn>,
        selector: Value,
    ) -> TCResult<CollectionItem<Self::Item, Self::Slice>> {
        TensorView::from(self.clone()).get(txn, selector).await
    }

    async fn is_empty(&self, txn: Arc<Txn>) -> TCResult<bool> {
        TensorView::from(self.clone()).is_empty(txn).await
    }

    async fn put(
        &self,
        txn: Arc<Txn>,
        selector: Value,
        value: CollectionItem<Self::Item, Self::Slice>,
    ) -> TCResult<()> {
        TensorView::from(self.clone())
            .put(txn, selector, value)
            .await
    }

    async fn to_stream(&self, txn: Arc<Txn>) -> TCResult<TCStream<Value>> {
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
    type Item = Number;
    type Slice = TensorView;

    async fn get(
        &self,
        txn: Arc<Txn>,
        selector: Value,
    ) -> TCResult<CollectionItem<Self::Item, Self::Slice>> {
        let bounds: Bounds = selector.try_into()?;
        if bounds.is_coord() {
            let coord: Vec<u64> = bounds.try_into()?;
            let value = self.read_value(&txn, &coord).await?;
            Ok(CollectionItem::Value(value))
        } else {
            let slice = self.slice(bounds)?;
            Ok(CollectionItem::Slice(slice))
        }
    }

    async fn is_empty(&self, txn: Arc<Txn>) -> TCResult<bool> {
        self.any(txn).map_ok(|any| !any).await
    }

    async fn put(
        &self,
        txn: Arc<Txn>,
        selector: Value,
        value: CollectionItem<Self::Item, Self::Slice>,
    ) -> TCResult<()> {
        let bounds: Bounds = selector.try_into()?;
        match value {
            CollectionItem::Value(value) => self.write_value(txn.id().clone(), bounds, value).await,
            CollectionItem::Slice(slice) => self.write(txn, bounds, slice).await,
        }
    }

    async fn to_stream(&self, _txn: Arc<Txn>) -> TCResult<TCStream<Value>> {
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
    Base(TensorBaseType),
    View(TensorViewType),
}

impl Class for TensorType {
    type Instance = Tensor;

    fn from_path(path: &TCPath) -> TCResult<Self> {
        TensorBaseType::from_path(path).map(TensorType::Base)
    }

    fn prefix() -> TCPath {
        TensorBaseType::prefix()
    }
}

impl From<TensorType> for Link {
    fn from(tt: TensorType) -> Link {
        match tt {
            TensorType::Base(base) => base.into(),
            TensorType::View(view) => view.into(),
        }
    }
}

impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Base(base) => write!(f, "{}", base),
            Self::View(view) => write!(f, "{}", view),
        }
    }
}

#[derive(Clone)]
pub enum Tensor {
    Base(TensorBase),
    View(TensorView),
}

impl Instance for Tensor {
    type Class = TensorType;

    fn class(&self) -> Self::Class {
        match self {
            Self::Base(base) => TensorType::Base(base.class()),
            Self::View(view) => TensorType::View(view.class()),
        }
    }
}

#[async_trait]
impl CollectionInstance for Tensor {
    type Item = Number;
    type Slice = TensorView;

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

    async fn to_stream(&self, txn: Arc<Txn>) -> TCResult<TCStream<Value>> {
        match self {
            Self::Base(base) => base.to_stream(txn).await,
            Self::View(view) => view.to_stream(txn).await,
        }
    }
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

impl From<Tensor> for Collection {
    fn from(tensor: Tensor) -> Collection {
        match tensor {
            Tensor::Base(base) => Collection::Base(CollectionBase::Tensor(base)),
            Tensor::View(view) => Collection::View(CollectionView::Tensor(view)),
        }
    }
}
