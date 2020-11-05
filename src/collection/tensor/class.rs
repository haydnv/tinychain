use std::convert::{TryFrom, TryInto};
use std::fmt;

use async_trait::async_trait;
use futures::stream::{self, StreamExt};
use futures::TryFutureExt;

use crate::class::{Class, Instance, NativeClass, TCResult, TCStream};
use crate::collection::class::*;
use crate::collection::{
    Collection, CollectionBase, CollectionBaseType, CollectionType, CollectionView,
};
use crate::error;
use crate::scalar::*;
use crate::transaction::{Transact, Txn, TxnId};

use super::bounds::{Bounds, Shape};
use super::dense::BlockListFile;
use super::sparse::SparseTable;
use super::{DenseTensor, SparseTensor, TensorBoolean, TensorIO, TensorTransform};

const ERR_CREATE_DENSE: &str = "DenseTensor can be constructed with (NumberType, Shape) or \
(Number, ...), not";

pub trait TensorInstance: Send {
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
}

impl NativeClass for TensorBaseType {
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

    async fn get(&self, txn: &Txn, schema: Value) -> TCResult<TensorBase> {
        match self {
            Self::Dense => {
                if schema.matches::<(NumberType, Shape)>() {
                    let (dtype, shape): (NumberType, Shape) = schema.opt_cast_into().unwrap();
                    let block_list = BlockListFile::constant(txn, shape, dtype.zero()).await?;
                    Ok(TensorBase::Dense(block_list))
                } else if schema.matches::<Vec<Number>>() {
                    let mut data: Vec<Number> = schema.opt_cast_into().unwrap();
                    let shape = vec![data.len() as u64].into();
                    let dtype = data
                        .iter()
                        .map(|n| n.class())
                        .fold(NumberType::Bool, Ord::max);
                    let block_list =
                        BlockListFile::from_values(txn, shape, dtype, stream::iter(data.drain(..)))
                            .await?;
                    Ok(TensorBase::Dense(block_list))
                } else {
                    Err(error::bad_request(ERR_CREATE_DENSE, schema))
                }
            }
            Self::Sparse => todo!(),
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
        txn: Txn,
        path: TCPath,
        selector: Value,
    ) -> TCResult<CollectionItem<Self::Item, Self::Slice>> {
        TensorView::from(self.clone())
            .get(txn, path, selector)
            .await
    }

    async fn is_empty(&self, txn: &Txn) -> TCResult<bool> {
        TensorView::from(self.clone()).is_empty(txn).await
    }

    async fn put(
        &self,
        txn: Txn,
        path: TCPath,
        selector: Value,
        value: CollectionItem<Self::Item, Self::Slice>,
    ) -> TCResult<()> {
        TensorView::from(self.clone())
            .put(txn, path, selector, value)
            .await
    }

    async fn to_stream(&self, txn: Txn) -> TCResult<TCStream<Scalar>> {
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

    async fn finalize(&self, txn_id: &TxnId) {
        match self {
            Self::Dense(dense) => dense.finalize(txn_id).await,
            Self::Sparse(sparse) => sparse.finalize(txn_id).await,
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
}

impl NativeClass for TensorViewType {
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
        txn: Txn,
        path: TCPath,
        selector: Value,
    ) -> TCResult<CollectionItem<Self::Item, Self::Slice>> {
        if !path.is_empty() {
            return Err(error::not_found(path));
        }

        let bounds: Bounds = selector
            .try_cast_into(|s| error::bad_request("Expected Tensor bounds but found", s))?;

        if bounds.is_coord() {
            let coord: Vec<u64> = bounds.try_into()?;
            let value = self.read_value(&txn, &coord).await?;
            Ok(CollectionItem::Scalar(value))
        } else {
            let slice = self.slice(bounds)?;
            Ok(CollectionItem::Slice(slice))
        }
    }

    async fn is_empty(&self, txn: &Txn) -> TCResult<bool> {
        self.any(txn.clone()).map_ok(|any| !any).await
    }

    async fn put(
        &self,
        txn: Txn,
        path: TCPath,
        selector: Value,
        value: CollectionItem<Self::Item, Self::Slice>,
    ) -> TCResult<()> {
        if !path.is_empty() {
            return Err(error::not_found(path));
        }

        let bounds: Bounds = selector
            .try_cast_into(|s| error::bad_request("Expected Tensor bounds but found", s))?;

        match value {
            CollectionItem::Scalar(value) => {
                self.write_value(txn.id().clone(), bounds, value).await
            }
            CollectionItem::Slice(slice) => self.write(txn, bounds, slice).await,
        }
    }

    async fn to_stream(&self, txn: Txn) -> TCResult<TCStream<Scalar>> {
        match self {
            // TODO: Forward errors, don't panic!
            Self::Dense(dense) => {
                let result_stream = dense.value_stream(txn).await?;
                let values: TCStream<Scalar> = Box::pin(
                    result_stream.map(|r| r.map(Value::Number).map(Scalar::Value).unwrap()),
                );
                Ok(values)
            }
            Self::Sparse(sparse) => {
                let result_stream = sparse.filled(txn).await?;
                let values: TCStream<Scalar> = Box::pin(
                    result_stream
                        .map(|r| r.unwrap())
                        .map(Value::from)
                        .map(Scalar::Value),
                );
                Ok(values)
            }
        }
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

    async fn finalize(&self, txn_id: &TxnId) {
        match self {
            Self::Dense(dense) => dense.finalize(txn_id).await,
            Self::Sparse(sparse) => sparse.finalize(txn_id).await,
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

impl TryFrom<CollectionView> for TensorView {
    type Error = error::TCError;

    fn try_from(view: CollectionView) -> TCResult<TensorView> {
        match view {
            CollectionView::Tensor(tensor) => match tensor {
                Tensor::Base(tb) => Ok(tb.into()),
                Tensor::View(tv) => Ok(tv),
            },
            other => Err(error::bad_request("Expected TensorView but found", other)),
        }
    }
}

impl From<TensorView> for Collection {
    fn from(view: TensorView) -> Collection {
        Collection::View(CollectionView::Tensor(view.into()))
    }
}

#[derive(Clone, Eq, PartialEq)]
pub enum TensorType {
    Base(TensorBaseType),
    View(TensorViewType),
}

impl Class for TensorType {
    type Instance = Tensor;
}

impl NativeClass for TensorType {
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

    async fn finalize(&self, txn_id: &TxnId) {
        match self {
            Self::Base(base) => base.finalize(txn_id).await,
            Self::View(view) => view.finalize(txn_id).await,
        }
    }
}

impl From<TensorBase> for Tensor {
    fn from(tb: TensorBase) -> Tensor {
        Tensor::Base(tb)
    }
}

impl From<TensorView> for Tensor {
    fn from(tv: TensorView) -> Tensor {
        Tensor::View(tv)
    }
}

impl From<Tensor> for Collection {
    fn from(tensor: Tensor) -> Collection {
        match tensor {
            Tensor::Base(base) => Collection::Base(CollectionBase::Tensor(base)),
            Tensor::View(view) => Collection::View(CollectionView::Tensor(view.into())),
        }
    }
}
