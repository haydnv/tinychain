use std::convert::{TryFrom, TryInto};
use std::fmt;

use async_trait::async_trait;
use futures::stream::StreamExt;
use futures::TryFutureExt;

use crate::class::{Class, Instance, NativeClass, State, TCResult, TCStream};
use crate::collection::class::*;
use crate::collection::{
    Collection, CollectionBase, CollectionBaseType, CollectionType, CollectionView,
};
use crate::error;
use crate::request::Request;
use crate::scalar::*;
use crate::transaction::{Transact, Txn, TxnId};

use super::bounds::{Bounds, Shape};
use super::dense::BlockListFile;
use super::sparse::SparseTable;
use super::{DenseTensor, SparseTensor, TensorBoolean, TensorIO, TensorTransform};

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
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        let suffix = Self::prefix().try_suffix(path)?;

        if suffix.len() == 1 {
            match suffix[0].as_str() {
                "dense" => Ok(TensorBaseType::Dense),
                "sparse" => Ok(TensorBaseType::Sparse),
                other => Err(error::not_found(other)),
            }
        } else {
            Err(error::path_not_found(suffix))
        }
    }

    fn prefix() -> TCPathBuf {
        CollectionBaseType::prefix().append(label("tensor"))
    }
}

#[async_trait]
impl CollectionClass for TensorBaseType {
    type Instance = TensorBase;

    async fn get(&self, txn: &Txn, schema: Value) -> TCResult<TensorBase> {
        let (dtype, shape): (NumberType, Shape) = schema.try_cast_into(|v| {
            error::bad_request("Tensor schema is (NumberType, Shape), not", v)
        })?;

        match self {
            Self::Dense => {
                BlockListFile::constant(txn, shape, dtype.zero())
                    .map_ok(TensorBase::Dense)
                    .await
            }
            Self::Sparse => {
                SparseTable::create(txn, shape, dtype)
                    .map_ok(TensorBase::Sparse)
                    .await
            }
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
            Dense => prefix.append(label("dense")).into(),
            Sparse => prefix.append(label("sparse")).into(),
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
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        selector: Value,
    ) -> TCResult<State> {
        TensorView::from(self.clone())
            .get(request, txn, path, selector)
            .await
    }

    async fn is_empty(&self, txn: &Txn) -> TCResult<bool> {
        TensorView::from(self.clone()).is_empty(txn).await
    }

    async fn post(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        params: Object,
    ) -> TCResult<State> {
        TensorView::from(self.clone())
            .post(request, txn, path, params)
            .await
    }

    async fn put(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        selector: Value,
        value: State,
    ) -> TCResult<()> {
        TensorView::from(self.clone())
            .put(request, txn, path, selector, value)
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
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        Err(error::bad_request(
            crate::class::ERR_PROTECTED,
            TCPath::from(path),
        ))
    }

    fn prefix() -> TCPathBuf {
        TensorBaseType::prefix()
    }
}

impl From<TensorViewType> for Link {
    fn from(tvt: TensorViewType) -> Link {
        let prefix = TensorViewType::prefix();

        use TensorViewType::*;
        match tvt {
            Dense => prefix.append(label("dense")).into(),
            Sparse => prefix.append(label("sparse")).into(),
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
        _request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        selector: Value,
    ) -> TCResult<State> {
        if !path.is_empty() {
            return Err(error::path_not_found(path));
        }

        let bounds: Bounds = selector
            .try_cast_into(|s| error::bad_request("Expected Tensor bounds but found", s))?;

        if bounds.is_coord() {
            let coord: Vec<u64> = bounds.try_into()?;
            let value = self.read_value(&txn, &coord).await?;
            Ok(State::Scalar(Scalar::Value(Value::Number(value))))
        } else {
            let slice = self.slice(bounds)?;
            Ok(State::Collection(slice.into()))
        }
    }

    async fn is_empty(&self, txn: &Txn) -> TCResult<bool> {
        self.any(txn.clone()).map_ok(|any| !any).await
    }

    async fn post(
        &self,
        _request: &Request,
        _txn: &Txn,
        _path: &[PathSegment],
        _params: Object,
    ) -> TCResult<State> {
        Err(error::not_implemented("TensorView::post"))
    }

    async fn put(
        &self,
        _request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        selector: Value,
        value: State,
    ) -> TCResult<()> {
        if !path.is_empty() {
            return Err(error::path_not_found(path));
        }

        let bounds: Bounds = selector
            .try_cast_into(|s| error::bad_request("Expected Tensor bounds but found", s))?;

        match value {
            State::Scalar(Scalar::Value(Value::Number(value))) => {
                self.write_value(txn.id().clone(), bounds, value).await
            }
            State::Collection(Collection::Base(CollectionBase::Tensor(tensor))) => {
                self.write(txn.clone(), bounds, tensor.into()).await
            }
            State::Collection(Collection::View(CollectionView::Tensor(tensor))) => {
                self.write(txn.clone(), bounds, tensor.into()).await
            }
            other => Err(error::bad_request(
                "Not a valid Tensor value or slice",
                other,
            )),
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
    fn from_path(path: &[PathSegment]) -> TCResult<Self> {
        TensorBaseType::from_path(path).map(TensorType::Base)
    }

    fn prefix() -> TCPathBuf {
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
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        selector: Value,
    ) -> TCResult<State> {
        match self {
            Self::Base(base) => base.get(request, txn, path, selector).await,
            Self::View(view) => view.get(request, txn, path, selector).await,
        }
    }

    async fn is_empty(&self, txn: &Txn) -> TCResult<bool> {
        match self {
            Self::Base(base) => base.is_empty(txn).await,
            Self::View(view) => view.is_empty(txn).await,
        }
    }

    async fn post(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        params: Object,
    ) -> TCResult<State> {
        match self {
            Self::Base(base) => base.post(request, txn, path, params).await,
            Self::View(view) => view.post(request, txn, path, params).await,
        }
    }

    async fn put(
        &self,
        request: &Request,
        txn: &Txn,
        path: &[PathSegment],
        selector: Value,
        value: State,
    ) -> TCResult<()> {
        match self {
            Self::Base(base) => base.put(request, txn, path, selector, value).await,
            Self::View(view) => view.put(request, txn, path, selector, value).await,
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

impl From<Tensor> for TensorView {
    fn from(tensor: Tensor) -> TensorView {
        match tensor {
            Tensor::Base(base) => base.into(),
            Tensor::View(view) => view,
        }
    }
}
