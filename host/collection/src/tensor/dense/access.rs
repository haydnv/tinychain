use std::cmp::Ordering;
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use collate::Collate;
use destream::de;
use freqfs::*;
use futures::future::{Future, FutureExt, TryFutureExt};
use futures::stream::{Stream, StreamExt, TryStreamExt};
use futures::try_join;
use ha_ndarray::*;
use log::{debug, trace};
use safecast::{AsType, CastFrom, CastInto};
use smallvec::{smallvec, SmallVec};

use tc_error::*;
use tc_transact::lock::{PermitRead, PermitWrite};
use tc_transact::{fs, Transaction, TxnId};
use tc_value::{
    DType, Number, NumberClass, NumberCollator, NumberInstance, NumberType, Trigonometry,
};
use tcgeneric::ThreadSafe;

use super::base::DenseBase;
use super::file::DenseFile;
use super::{
    block_axis_for, block_map_for, block_shape_for, div_ceil, ideal_block_size_for, DenseInstance,
    DenseWrite, DenseWriteGuard, DenseWriteLock,
};

use crate::tensor::block::Block;
use crate::tensor::sparse::{Node, SparseAccess, SparseInstance};
use crate::tensor::transform::{Broadcast, Expand, Reduce, Reshape, Slice, Transpose};
use crate::tensor::{
    autoqueue, coord_of, offset_of, Axes, AxisRange, Coord, Range, Semaphore, Shape,
    TensorInstance, TensorPermitRead, TensorPermitWrite,
};

use super::stream::{BlockResize, ValueStream};
use super::{BlockShape, BlockStream, DenseCacheFile};

pub enum DenseAccessCast<Txn, FE> {
    F32(DenseAccess<Txn, FE, f32>),
    F64(DenseAccess<Txn, FE, f64>),
    I16(DenseAccess<Txn, FE, i16>),
    I32(DenseAccess<Txn, FE, i32>),
    I64(DenseAccess<Txn, FE, i64>),
    U8(DenseAccess<Txn, FE, u8>),
    U16(DenseAccess<Txn, FE, u16>),
    U32(DenseAccess<Txn, FE, u32>),
    U64(DenseAccess<Txn, FE, u64>),
}

macro_rules! cast_dispatch {
    ($this:ident, $var:ident, $call:expr) => {
        match $this {
            DenseAccessCast::F32($var) => $call,
            DenseAccessCast::F64($var) => $call,
            DenseAccessCast::I16($var) => $call,
            DenseAccessCast::I32($var) => $call,
            DenseAccessCast::I64($var) => $call,
            DenseAccessCast::U8($var) => $call,
            DenseAccessCast::U16($var) => $call,
            DenseAccessCast::U32($var) => $call,
            DenseAccessCast::U64($var) => $call,
        }
    };
}

impl<Txn, FE> Clone for DenseAccessCast<Txn, FE> {
    fn clone(&self) -> Self {
        match self {
            Self::F32(access) => Self::F32(access.clone()),
            Self::F64(access) => Self::F64(access.clone()),
            Self::I16(access) => Self::I16(access.clone()),
            Self::I32(access) => Self::I32(access.clone()),
            Self::I64(access) => Self::I64(access.clone()),
            Self::U8(access) => Self::U8(access.clone()),
            Self::U16(access) => Self::U16(access.clone()),
            Self::U32(access) => Self::U32(access.clone()),
            Self::U64(access) => Self::U64(access.clone()),
        }
    }
}

impl<Txn, FE> DenseAccessCast<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node>,
{
    async fn read_block(&self, txn_id: TxnId, block_id: u64) -> TCResult<Block> {
        cast_dispatch!(
            self,
            this,
            this.read_block(txn_id, block_id).map_ok(Block::from).await
        )
    }

    async fn read_blocks(
        self,
        txn_id: TxnId,
    ) -> TCResult<Pin<Box<dyn Stream<Item = TCResult<Block>> + Send>>> {
        cast_dispatch!(self, this, {
            let blocks = this.read_blocks(txn_id).await?;
            let blocks = blocks.map_ok(Block::from);
            Ok(Box::pin(blocks))
        })
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> TCResult<Number> {
        cast_dispatch!(
            self,
            this,
            this.read_value(txn_id, coord).map_ok(Number::from).await
        )
    }
}

macro_rules! access_cast_from {
    ($t:ty, $var:ident) => {
        impl<Txn, FE> From<DenseAccess<Txn, FE, $t>> for DenseAccessCast<Txn, FE> {
            fn from(access: DenseAccess<Txn, FE, $t>) -> Self {
                Self::$var(access)
            }
        }
    };
}

access_cast_from!(f32, F32);
access_cast_from!(f64, F64);
access_cast_from!(i16, I16);
access_cast_from!(i32, I32);
access_cast_from!(i64, I64);
access_cast_from!(u8, U8);
access_cast_from!(u16, U16);
access_cast_from!(u32, U32);
access_cast_from!(u64, U64);

#[async_trait]
impl<Txn: ThreadSafe, FE: ThreadSafe> TensorPermitRead for DenseAccessCast<Txn, FE> {
    async fn read_permit(
        &self,
        txn_id: TxnId,
        range: Range,
    ) -> TCResult<SmallVec<[PermitRead<Range>; 16]>> {
        cast_dispatch!(self, this, this.read_permit(txn_id, range).await)
    }
}

impl<Txn, FE> fmt::Debug for DenseAccessCast<Txn, FE>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        cast_dispatch!(self, this, this.fmt(f))
    }
}

pub enum DenseAccess<Txn, FE, T: CType> {
    Base(DenseBase<Txn, FE, T>),
    File(DenseFile<FE, T>),
    Broadcast(Box<DenseBroadcast<Self>>),
    Combine(Box<DenseCombine<Self, Self, T>>),
    CombineConst(Box<DenseCombineConst<Self, T>>),
    Compare(Box<DenseCompare<Txn, FE, T>>),
    CompareConst(Box<DenseCompareConst<Txn, FE, T>>),
    Cond(Box<DenseCond<DenseAccess<Txn, FE, u8>, Self, Self>>),
    Const(Box<DenseConst<Self, T>>),
    Cow(Box<DenseCow<FE, Self>>),
    Diagonal(Box<DenseDiagonal<Self>>),
    Expand(Box<DenseExpand<Self>>),
    MatMul(Box<DenseMatMul<Self, Self>>),
    Reduce(Box<DenseReduce<Self, T>>),
    Reshape(Box<DenseReshape<Self>>),
    ResizeBlocks(Box<DenseResizeBlocks<Self>>),
    Slice(Box<DenseSlice<Self>>),
    Sparse(Box<DenseSparse<SparseAccess<Txn, FE, T>>>),
    Transpose(Box<DenseTranspose<Self>>),
    Unary(Box<DenseUnary<Self, T>>),
    UnaryCast(Box<DenseUnaryCast<Txn, FE, T>>),
    Version(DenseVersion<FE, T>),
}

impl<Txn, FE, T: CType> Clone for DenseAccess<Txn, FE, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Base(base) => Self::Base(base.clone()),
            Self::File(file) => Self::File(file.clone()),
            Self::Broadcast(broadcast) => Self::Broadcast(broadcast.clone()),
            Self::Combine(combine) => Self::Combine(combine.clone()),
            Self::CombineConst(combine) => Self::CombineConst(combine.clone()),
            Self::Compare(compare) => Self::Compare(compare.clone()),
            Self::CompareConst(compare) => Self::CompareConst(compare.clone()),
            Self::Cond(cond) => Self::Cond(cond.clone()),
            Self::Const(combine) => Self::Const(combine.clone()),
            Self::Cow(cow) => Self::Cow(cow.clone()),
            Self::Diagonal(diag) => Self::Diagonal(diag.clone()),
            Self::MatMul(matmul) => Self::MatMul(matmul.clone()),
            Self::Expand(expand) => Self::Expand(expand.clone()),
            Self::Reduce(reduce) => Self::Reduce(reduce.clone()),
            Self::Reshape(reshape) => Self::Reshape(reshape.clone()),
            Self::ResizeBlocks(resize) => Self::ResizeBlocks(resize.clone()),
            Self::Slice(slice) => Self::Slice(slice.clone()),
            Self::Sparse(sparse) => Self::Sparse(sparse.clone()),
            Self::Transpose(transpose) => Self::Transpose(transpose.clone()),
            Self::Unary(unary) => Self::Unary(unary.clone()),
            Self::UnaryCast(unary) => Self::UnaryCast(unary.clone()),
            Self::Version(version) => Self::Version(version.clone()),
        }
    }
}

macro_rules! access_dispatch {
    ($this:ident, $var:ident, $call:expr) => {
        match $this {
            DenseAccess::Base($var) => $call,
            DenseAccess::File($var) => $call,
            DenseAccess::Broadcast($var) => $call,
            DenseAccess::Combine($var) => $call,
            DenseAccess::CombineConst($var) => $call,
            DenseAccess::Compare($var) => $call,
            DenseAccess::CompareConst($var) => $call,
            DenseAccess::Cond($var) => $call,
            DenseAccess::Const($var) => $call,
            DenseAccess::Cow($var) => $call,
            DenseAccess::Diagonal($var) => $call,
            DenseAccess::Expand($var) => $call,
            DenseAccess::MatMul($var) => $call,
            DenseAccess::Reduce($var) => $call,
            DenseAccess::Reshape($var) => $call,
            DenseAccess::ResizeBlocks($var) => $call,
            DenseAccess::Slice($var) => $call,
            DenseAccess::Sparse($var) => $call,
            DenseAccess::Transpose($var) => $call,
            DenseAccess::Unary($var) => $call,
            DenseAccess::UnaryCast($var) => $call,
            DenseAccess::Version($var) => $call,
        }
    };
}

impl<Txn, FE, T> TensorInstance for DenseAccess<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CType + DType,
{
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        access_dispatch!(self, this, this.shape())
    }
}

#[async_trait]
impl<Txn, FE, T> DenseInstance for DenseAccess<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Buffer<T>> + AsType<Node>,
    T: CType + DType + fmt::Debug,
    Buffer<T>: de::FromStream<Context = ()>,
    Number: From<T> + CastInto<T>,
{
    type Block = Array<T>;
    type DType = T;

    fn block_size(&self) -> usize {
        access_dispatch!(self, this, this.block_size())
    }

    async fn read_block(&self, txn_id: TxnId, block_id: u64) -> TCResult<Self::Block> {
        access_dispatch!(
            self,
            this,
            this.read_block(txn_id, block_id).map_ok(Array::from).await
        )
    }

    async fn read_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::Block>> {
        match self {
            Self::Base(base) => base.read_blocks(txn_id).await,
            Self::File(file) => Ok(Box::pin(
                file.read_blocks(txn_id).await?.map_ok(Array::from),
            )),
            Self::Broadcast(broadcast) => Ok(Box::pin(
                broadcast.read_blocks(txn_id).await?.map_ok(Array::from),
            )),
            Self::Combine(combine) => combine.read_blocks(txn_id).await,
            Self::CombineConst(combine) => combine.read_blocks(txn_id).await,
            Self::Compare(compare) => compare.read_blocks(txn_id).await,
            Self::CompareConst(compare) => compare.read_blocks(txn_id).await,
            Self::Cond(cond) => Ok(Box::pin(
                cond.read_blocks(txn_id).await?.map_ok(Array::from),
            )),
            Self::Const(combine) => combine.read_blocks(txn_id).await,
            Self::Cow(cow) => cow.read_blocks(txn_id).await,
            Self::Diagonal(diag) => Ok(Box::pin(
                diag.read_blocks(txn_id).await?.map_ok(Array::from),
            )),
            Self::Expand(expand) => Ok(Box::pin(
                expand.read_blocks(txn_id).await?.map_ok(Array::from),
            )),
            Self::MatMul(matmul) => Ok(Box::pin(
                matmul.read_blocks(txn_id).await?.map_ok(Array::from),
            )),
            Self::Reduce(reduce) => reduce.read_blocks(txn_id).await,
            Self::Reshape(reshape) => Ok(Box::pin(
                reshape.read_blocks(txn_id).await?.map_ok(Array::from),
            )),
            Self::ResizeBlocks(resize) => Ok(Box::pin(
                resize.read_blocks(txn_id).await?.map_ok(Array::from),
            )),
            Self::Slice(slice) => Ok(Box::pin(
                slice.read_blocks(txn_id).await?.map_ok(Array::from),
            )),
            Self::Sparse(sparse) => Ok(Box::pin(
                sparse.read_blocks(txn_id).await?.map_ok(Array::from),
            )),
            Self::Transpose(transpose) => Ok(Box::pin(
                transpose.read_blocks(txn_id).await?.map_ok(Array::from),
            )),
            Self::Unary(unary) => unary.read_blocks(txn_id).await,
            Self::UnaryCast(unary) => unary.read_blocks(txn_id).await,
            Self::Version(version) => Ok(Box::pin(
                version.read_blocks(txn_id).await?.map_ok(Array::from),
            )),
        }
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> TCResult<Self::DType> {
        access_dispatch!(self, this, this.read_value(txn_id, coord).await)
    }
}

#[async_trait]
impl<Txn, FE, T> TensorPermitRead for DenseAccess<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CType + DType,
{
    async fn read_permit(
        &self,
        txn_id: TxnId,
        range: Range,
    ) -> TCResult<SmallVec<[PermitRead<Range>; 16]>> {
        match self {
            Self::Base(base) => base.read_permit(txn_id, range).await,
            Self::Broadcast(broadcast) => broadcast.read_permit(txn_id, range).await,
            Self::Combine(combine) => combine.read_permit(txn_id, range).await,
            Self::CombineConst(combine) => combine.read_permit(txn_id, range).await,
            Self::Compare(compare) => compare.read_permit(txn_id, range).await,
            Self::CompareConst(compare) => compare.read_permit(txn_id, range).await,
            Self::Cond(cond) => cond.read_permit(txn_id, range).await,
            Self::Const(combine) => combine.read_permit(txn_id, range).await,
            Self::Cow(cow) => cow.read_permit(txn_id, range).await,
            Self::Diagonal(diag) => diag.read_permit(txn_id, range).await,
            Self::Expand(expand) => expand.read_permit(txn_id, range).await,
            Self::MatMul(matmul) => matmul.read_permit(txn_id, range).await,
            Self::Reduce(reduce) => reduce.read_permit(txn_id, range).await,
            Self::Reshape(reshape) => reshape.read_permit(txn_id, range).await,
            Self::ResizeBlocks(resize) => resize.read_permit(txn_id, range).await,
            Self::Slice(slice) => slice.read_permit(txn_id, range).await,
            Self::Sparse(sparse) => sparse.read_permit(txn_id, range).await,
            Self::Transpose(transpose) => transpose.read_permit(txn_id, range).await,
            Self::Unary(unary) => unary.read_permit(txn_id, range).await,
            Self::UnaryCast(unary) => unary.read_permit(txn_id, range).await,
            Self::Version(version) => version.read_permit(txn_id, range).await,

            other => Err(bad_request!(
                "{:?} does not support transactional locking",
                other
            )),
        }
    }
}

#[async_trait]
impl<Txn, FE, T> TensorPermitWrite for DenseAccess<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CType + DType,
{
    async fn write_permit(&self, txn_id: TxnId, range: Range) -> TCResult<PermitWrite<Range>> {
        match self {
            Self::Base(base) => base.write_permit(txn_id, range).await,
            Self::Slice(slice) => slice.write_permit(txn_id, range).await,
            Self::Version(version) => version.write_permit(txn_id, range).await,
            other => Err(bad_request!(
                "{:?} does not support transactional writes",
                other
            )),
        }
    }
}

impl<Txn, FE, T: CType + DType> fmt::Debug for DenseAccess<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CType + DType,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        access_dispatch!(self, this, this.fmt(f))
    }
}

#[derive(Clone)]
pub struct DenseBroadcast<S> {
    source: S,
    transform: Broadcast,
    block_map: ArrayBuf<StackVec<u64>>,
    block_size: usize,
}

impl<S: DenseInstance> DenseBroadcast<S> {
    pub fn new(source: S, shape: Shape) -> TCResult<Self> {
        debug!(
            "broadcast {source:?} with block size {block_size} into {shape:?}",
            block_size = source.block_size()
        );

        let transform = Broadcast::new(source.shape().clone(), shape)?;

        // characterize the source tensor
        let num_blocks = source.size() / source.block_size() as u64;
        let block_axis = block_axis_for(source.shape(), source.block_size());
        let source_block_shape = block_shape_for(block_axis, source.shape(), source.block_size());
        let source_block_map = block_map_for(num_blocks, source.shape(), &source_block_shape)?;

        // characterize the output tensor (this tensor)
        let axis_offset = transform.shape().len() - source.ndim();
        let block_axis = block_axis_for(
            transform.shape(),
            (transform.shape().size() / num_blocks) as usize,
        );

        let map_shape = transform
            .shape()
            .iter()
            .take(axis_offset + block_axis)
            .copied()
            .map(|dim| dim as usize)
            .collect::<Vec<usize>>();

        let block_shape = transform
            .shape()
            .iter()
            .skip(axis_offset + block_axis)
            .copied()
            .map(|dim| dim as usize)
            .collect::<Vec<usize>>();

        let block_map = if map_shape.is_empty() {
            source_block_map
        } else if source_block_map.size() == map_shape.iter().product::<usize>() {
            // TODO: this should not be necessary
            let block_map = source_block_map.reshape(map_shape)?;
            ArrayBuf::copy(&block_map)?
        } else {
            let block_map = source_block_map.broadcast(map_shape)?;
            ArrayBuf::copy(&block_map)?
        };

        Ok(Self {
            source,
            transform,
            block_map,
            block_size: block_shape.iter().product(),
        })
    }
}

impl<S: TensorInstance> TensorInstance for DenseBroadcast<S> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
        self.transform.shape()
    }
}

#[async_trait]
impl<S: DenseInstance + Clone> DenseInstance for DenseBroadcast<S>
where
    S::Block: NDArrayTransform,
    <S::Block as NDArrayTransform>::Broadcast:
        NDArrayRead<DType = S::DType> + NDArrayTransform + Into<Array<S::DType>>,
{
    type Block = <S::Block as NDArrayTransform>::Broadcast;
    type DType = S::DType;

    fn block_size(&self) -> usize {
        self.block_size
    }

    async fn read_block(&self, txn_id: TxnId, block_id: u64) -> TCResult<Self::Block> {
        let source_block_id = source_block_id_for(&self.block_map, block_id)?;
        let block_axis = block_axis_for(self.shape(), self.block_size);
        let block_shape = block_shape_for(block_axis, self.shape(), self.block_size);

        let this = &self;
        debug!("DenseBroadcast::read_block {block_id} from {this:?} by broadcasting source block {source_block_id} into {block_shape:?}");

        let source_block = self.source.read_block(txn_id, source_block_id).await?;
        source_block.broadcast(block_shape).map_err(TCError::from)
    }

    async fn read_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::Block>> {
        let block_axis = block_axis_for(self.shape(), self.block_size);
        let block_shape = block_shape_for(block_axis, self.shape(), self.block_size);

        let this = &self;
        debug!("DenseBroadcast::read_blocks from {this:?} by broadcasting source blocks into {block_shape:?}");

        let blocks = futures::stream::iter(self.block_map.into_inner())
            .map(move |block_id| {
                let source = self.source.clone();
                async move { source.read_block(txn_id, block_id).await }
            })
            .buffered(num_cpus::get())
            .map(move |result| {
                let source_block = result?;

                trace!("broadcast {source_block:?} into {block_shape:?}");

                source_block
                    .broadcast(block_shape.to_vec())
                    .map_err(TCError::from)
            });

        Ok(Box::pin(blocks))
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> TCResult<Self::DType> {
        self.shape().validate_coord(&coord)?;
        let coord = self.transform.invert_coord(coord);
        self.source.read_value(txn_id, coord).await
    }
}

#[async_trait]
impl<S: TensorPermitRead> TensorPermitRead for DenseBroadcast<S> {
    async fn read_permit(
        &self,
        txn_id: TxnId,
        range: Range,
    ) -> TCResult<SmallVec<[PermitRead<Range>; 16]>> {
        self.transform.shape().validate_range(&range)?;
        let range = self.transform.invert_range(range);
        self.source.read_permit(txn_id, range).await
    }
}

impl<Txn, FE, T, S> From<DenseBroadcast<S>> for DenseAccess<Txn, FE, T>
where
    T: CType,
    S: Into<DenseAccess<Txn, FE, T>>,
{
    fn from(broadcast: DenseBroadcast<S>) -> Self {
        Self::Broadcast(Box::new(DenseBroadcast {
            source: broadcast.source.into(),
            transform: broadcast.transform.clone(),
            block_map: broadcast.block_map,
            block_size: broadcast.block_size,
        }))
    }
}

impl<S: fmt::Debug> fmt::Debug for DenseBroadcast<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "broadcast of {:?} into {:?}",
            self.source,
            self.transform.shape()
        )
    }
}

pub struct DenseCow<FE, S> {
    source: S,
    dir: DirLock<FE>,
}

impl<FE, S: Clone> Clone for DenseCow<FE, S> {
    fn clone(&self) -> Self {
        Self {
            source: self.source.clone(),
            dir: self.dir.clone(),
        }
    }
}

impl<FE, S> DenseCow<FE, S> {
    pub fn create(source: S, dir: DirLock<FE>) -> Self {
        Self { source, dir }
    }
}

impl<FE, S> DenseCow<FE, S>
where
    FE: DenseCacheFile + AsType<Buffer<S::DType>> + 'static,
    S: DenseInstance + Clone,
    Array<S::DType>: From<S::Block>,
    Buffer<S::DType>: de::FromStream<Context = ()>,
{
    async fn write_buffer(
        &self,
        txn_id: TxnId,
        block_id: u64,
    ) -> TCResult<FileWriteGuardOwned<FE, Buffer<S::DType>>> {
        let mut dir = self.dir.write().await;

        if let Some(buffer) = dir.get_file(&block_id) {
            buffer.write_owned().map_err(TCError::from).await
        } else {
            let block = self.source.read_block(txn_id, block_id).await?;
            let queue = autoqueue(&block)?;
            let buffer = block.read(&queue)?.into_buffer()?;

            let type_size = S::DType::dtype().size();
            let buffer_data_size = type_size * buffer.len();
            let buffer = dir.create_file(block_id.to_string(), buffer, buffer_data_size)?;

            buffer.into_write().map_err(TCError::from).await
        }
    }
}

impl<FE, S> TensorInstance for DenseCow<FE, S>
where
    FE: ThreadSafe,
    S: TensorInstance,
{
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
        self.source.shape()
    }
}

#[async_trait]
impl<FE, S> DenseInstance for DenseCow<FE, S>
where
    FE: DenseCacheFile + AsType<Buffer<S::DType>> + 'static,
    S: DenseInstance + Clone,
    Array<S::DType>: From<S::Block>,
    Buffer<S::DType>: de::FromStream<Context = ()>,
{
    type Block = Array<S::DType>;
    type DType = S::DType;

    fn block_size(&self) -> usize {
        self.source.block_size()
    }

    async fn read_block(&self, txn_id: TxnId, block_id: u64) -> TCResult<Self::Block> {
        debug!("DenseCow::read_block {block_id} at {txn_id}");

        let dir = self.dir.read().await;

        if let Some(block) = dir.get_file(&block_id) {
            debug!("DenseCow::read_block {block_id} at {txn_id} found new block");

            let buffer: Buffer<S::DType> = block
                .read_owned::<Buffer<S::DType>>()
                .map_ok(|block| block.clone().into())
                .map_err(TCError::from)
                .await?;

            let block_axis = block_axis_for(self.shape(), self.block_size());
            let block_shape = block_shape_for(block_axis, self.shape(), buffer.len());
            let block = ArrayBuf::<Buffer<S::DType>>::new(block_shape, buffer)?;

            Ok(block.into())
        } else {
            debug!(
                "DenseCow::read_block {block_id} at {txn_id} reading from {source:?}",
                source = self.source
            );

            self.source
                .read_block(txn_id, block_id)
                .map_ok(Array::from)
                .await
        }
    }

    async fn read_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::Block>> {
        let num_blocks = div_ceil(self.size(), self.block_size() as u64);

        let blocks = futures::stream::iter(0..num_blocks)
            .map(move |block_id| {
                let this = self.clone();
                async move { this.read_block(txn_id, block_id).await }
            })
            .buffered(num_cpus::get());

        Ok(Box::pin(blocks))
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> TCResult<Self::DType> {
        self.shape().validate_coord(&coord)?;

        let offset = offset_of(coord, self.shape());
        let block_id = offset / self.block_size() as u64;
        let block_offset = (offset % self.block_size() as u64) as usize;

        let block = self.read_block(txn_id, block_id).await?;
        let queue = autoqueue(&block)?;
        let buffer = block.read(&queue)?;
        Ok(buffer.to_slice()?.as_ref()[block_offset])
    }
}

#[async_trait]
impl<FE, S> TensorPermitRead for DenseCow<FE, S>
where
    FE: Send + Sync,
    S: TensorPermitRead,
{
    async fn read_permit(
        &self,
        txn_id: TxnId,
        range: Range,
    ) -> TCResult<SmallVec<[PermitRead<Range>; 16]>> {
        self.source.read_permit(txn_id, range).await
    }
}

#[async_trait]
impl<FE, S> TensorPermitWrite for DenseCow<FE, S>
where
    FE: Send + Sync,
    S: TensorPermitWrite,
{
    async fn write_permit(&self, txn_id: TxnId, range: Range) -> TCResult<PermitWrite<Range>> {
        self.source.write_permit(txn_id, range).await
    }
}

#[async_trait]
impl<FE, S> DenseWrite for DenseCow<FE, S>
where
    FE: DenseCacheFile + AsType<Buffer<S::DType>> + 'static,
    S: DenseInstance + Clone,
    Array<S::DType>: From<S::Block>,
    Buffer<S::DType>: de::FromStream<Context = ()>,
{
    type BlockWrite = ArrayBuf<FileWriteGuardOwned<FE, Buffer<S::DType>>>;

    async fn write_block(&self, txn_id: TxnId, block_id: u64) -> TCResult<Self::BlockWrite> {
        let buffer = self.write_buffer(txn_id, block_id).await?;
        let block_axis = block_axis_for(self.shape(), self.block_size());
        let block_shape = block_shape_for(block_axis, self.shape(), buffer.len());
        ArrayBuf::new(buffer, block_shape).map_err(TCError::from)
    }

    async fn write_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::BlockWrite>> {
        let num_blocks = div_ceil(self.size(), self.block_size() as u64);
        let blocks = futures::stream::iter(0..num_blocks).then(move |block_id| {
            let this = self.clone();
            async move { this.write_block(txn_id, block_id).await }
        });

        Ok(Box::pin(blocks))
    }
}

#[async_trait]
impl<'a, FE, S> DenseWriteLock<'a> for DenseCow<FE, S>
where
    FE: DenseCacheFile + AsType<Buffer<S::DType>> + 'static,
    S: DenseInstance + Clone,
    Array<S::DType>: From<S::Block>,
    Buffer<S::DType>: de::FromStream<Context = ()>,
{
    type WriteGuard = DenseCowWriteGuard<'a, FE, S>;

    async fn write(&'a self) -> Self::WriteGuard {
        DenseCowWriteGuard { cow: self }
    }
}

impl<'a, Txn, FE, S, T> From<DenseCow<FE, S>> for DenseAccess<Txn, FE, T>
where
    T: CType,
    DenseAccess<Txn, FE, T>: From<S>,
{
    fn from(cow: DenseCow<FE, S>) -> Self {
        Self::Cow(Box::new(DenseCow {
            source: cow.source.into(),
            dir: cow.dir,
        }))
    }
}

impl<FE, S: fmt::Debug> fmt::Debug for DenseCow<FE, S>
where
    FE: ThreadSafe,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "copy-on-write view of {:?}", self.source)
    }
}

pub struct DenseCowWriteGuard<'a, FE, S> {
    cow: &'a DenseCow<FE, S>,
}

#[async_trait]
impl<'a, FE, S> DenseWriteGuard<S::DType> for DenseCowWriteGuard<'a, FE, S>
where
    FE: DenseCacheFile + AsType<Buffer<S::DType>> + 'static,
    S: DenseInstance + Clone,
    Array<S::DType>: From<S::Block>,
    Buffer<S::DType>: de::FromStream<Context = ()>,
{
    async fn overwrite<O: DenseInstance<DType = S::DType>>(
        &self,
        txn_id: TxnId,
        other: O,
    ) -> TCResult<()> {
        let source = other.read_blocks(txn_id).await?;

        let block_axis = block_axis_for(self.cow.shape(), self.cow.block_size());
        let block_shape = block_shape_for(block_axis, self.cow.shape(), self.cow.block_size());
        let source = BlockResize::new(source, block_shape)?;

        let dest = self.cow.clone().write_blocks(txn_id).await?;

        dest.zip(source)
            .map(|(dest, source)| {
                let mut dest = dest?;
                let source = source?;
                dest.write(&source).map_err(TCError::from)
            })
            .try_fold((), |(), _| futures::future::ready(Ok(())))
            .await
    }

    async fn overwrite_value(&self, txn_id: TxnId, value: S::DType) -> TCResult<()> {
        let dest = self.cow.clone().write_blocks(txn_id).await?;
        dest.map_ok(|mut block| block.write_value(value))
            .try_fold((), |(), _| futures::future::ready(Ok(())))
            .await
    }

    async fn write_value(&self, txn_id: TxnId, coord: Coord, value: S::DType) -> TCResult<()> {
        self.cow.shape().validate_coord(&coord)?;

        let offset = offset_of(coord, self.cow.shape());
        let block_id = offset / self.cow.block_size() as u64;
        let block_offset = offset % self.cow.block_size() as u64;
        let mut buffer = self.cow.write_buffer(txn_id, block_id).await?;

        buffer
            .write_value_at(block_offset as usize, value)
            .map_err(TCError::from)
    }
}

#[derive(Clone)]
pub struct DenseDiagonal<S> {
    source: S,
    shape: Shape,
}

impl<S: TensorInstance + fmt::Debug> DenseDiagonal<S> {
    pub fn new(source: S) -> TCResult<Self> {
        if source.shape().len() >= 2
            && source.shape().iter().nth_back(0) == source.shape().iter().nth_back(1)
        {
            let mut shape = source.shape().clone();
            shape.pop();

            Ok(Self {
                source,
                shape: shape.into(),
            })
        } else {
            Err(bad_request!(
                "matrix diagonal requires a square matrix or batch of square matrices, not {:?}",
                source
            ))
        }
    }
}

impl<S: TensorInstance> TensorInstance for DenseDiagonal<S> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }
}

#[async_trait]
impl<S: DenseInstance> DenseInstance for DenseDiagonal<S> {
    type Block = ArrayOp<ha_ndarray::ops::MatDiag<S::Block>>;
    type DType = S::DType;

    fn block_size(&self) -> usize {
        let matrix_dim = self.shape.last().copied().expect("matrix dim") as usize;
        self.source.block_size() / matrix_dim
    }

    async fn read_block(&self, txn_id: TxnId, block_id: u64) -> TCResult<Self::Block> {
        self.source
            .read_block(txn_id, block_id)
            .map(|result| result.and_then(|block| block.diagonal().map_err(TCError::from)))
            .await
    }

    async fn read_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::Block>> {
        let source_blocks = self.source.read_blocks(txn_id).await?;
        let blocks = source_blocks
            .map(|result| result.and_then(|block| block.diagonal().map_err(TCError::from)));

        Ok(Box::pin(blocks))
    }

    async fn read_value(&self, txn_id: TxnId, mut coord: Coord) -> TCResult<Self::DType> {
        self.shape.validate_coord(&coord)?;
        let i = coord.last().copied().expect("i");
        coord.push(i);
        self.source.read_value(txn_id, coord).await
    }
}

#[async_trait]
impl<S: TensorInstance + TensorPermitRead + fmt::Debug> TensorPermitRead for DenseDiagonal<S> {
    async fn read_permit(
        &self,
        txn_id: TxnId,
        mut range: Range,
    ) -> TCResult<SmallVec<[PermitRead<Range>; 16]>> {
        let range = match range.len().cmp(&self.ndim()) {
            Ordering::Less => Ok(range),
            Ordering::Equal => {
                let axis_range = range.last().cloned().expect("last axis range");
                range.push(axis_range);
                Ok(range)
            }
            Ordering::Greater => Err(bad_request!("invalid range for {:?}: {:?}", self, range)),
        }?;

        self.source.read_permit(txn_id, range).await
    }
}

impl<Txn, FE, S, T> From<DenseDiagonal<S>> for DenseAccess<Txn, FE, T>
where
    S: Into<DenseAccess<Txn, FE, T>>,
    T: CType,
{
    fn from(diag: DenseDiagonal<S>) -> Self {
        Self::Diagonal(Box::new(DenseDiagonal {
            source: diag.source.into(),
            shape: diag.shape,
        }))
    }
}

impl<S: fmt::Debug> fmt::Debug for DenseDiagonal<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "matrix diagonal of {:?}", self.source)
    }
}

#[derive(Clone)]
pub struct DenseExpand<S> {
    source: S,
    transform: Expand,
}

impl<S: DenseInstance> DenseExpand<S> {
    pub fn new(source: S, axes: Axes) -> TCResult<Self> {
        Expand::new(source.shape().clone(), axes).map(|transform| Self { source, transform })
    }
}

impl<S: TensorInstance> TensorInstance for DenseExpand<S> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
        self.transform.shape()
    }
}

#[async_trait]
impl<S: DenseInstance + Clone> DenseInstance for DenseExpand<S> {
    type Block = Array<S::DType>;
    type DType = S::DType;

    fn block_size(&self) -> usize {
        self.source.block_size()
    }

    async fn read_block(&self, txn_id: TxnId, block_id: u64) -> TCResult<Self::Block> {
        debug!(
            "DenseExpand::read_block {block_id} from {:?} and expand axes {:?}",
            self.source,
            self.transform.expand_axes()
        );

        let block = self.source.read_block(txn_id, block_id).await?;
        assert!(
            block.ndim() <= self.source.ndim(),
            "{:?} returned a block with too many dimensions",
            self.source
        );

        let offset = self.source.ndim() - block.ndim();
        let axes = self
            .transform
            .expand_axes()
            .iter()
            .copied()
            .filter(|x| x >= &offset)
            .map(|x| x - offset)
            .collect::<Vec<usize>>();

        if axes.is_empty() {
            trace!("no need to expand {block:?}");
            Ok(block.into())
        } else {
            trace!("expand axes {axes:?} of {block:?}");
            block.into().expand_dims(axes).map_err(TCError::from)
        }
    }

    async fn read_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::Block>> {
        debug!(
            "DenseExpand::read_blocks from {:?} and expand axes {:?}",
            self.source,
            self.transform.expand_axes()
        );

        let source_ndim = self.source.ndim();
        let source_blocks = self.source.read_blocks(txn_id).await?;
        let transform = self.transform;

        let blocks = source_blocks.map(move |result| {
            let block = result?;
            assert!(block.ndim() <= source_ndim);

            let offset = source_ndim - block.ndim();
            let axes = transform
                .expand_axes()
                .iter()
                .copied()
                .filter(|x| *x >= offset)
                .map(|x| x - offset)
                .collect::<Vec<usize>>();

            if axes.is_empty() {
                trace!("no need to expand {block:?}");
                Ok(block.into())
            } else {
                trace!("expand axes {axes:?} of {block:?}");
                block.into().expand_dims(axes).map_err(TCError::from)
            }
        });

        Ok(Box::pin(blocks))
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> TCResult<Self::DType> {
        self.shape().validate_coord(&coord)?;
        let coord = self.transform.invert_coord(coord);
        self.source.read_value(txn_id, coord).await
    }
}

#[async_trait]
impl<S: TensorPermitRead + fmt::Debug> TensorPermitRead for DenseExpand<S> {
    async fn read_permit(
        &self,
        txn_id: TxnId,
        range: Range,
    ) -> TCResult<SmallVec<[PermitRead<Range>; 16]>> {
        self.transform.shape().validate_range(&range)?;
        let range = self.transform.invert_range(range);
        self.source.read_permit(txn_id, range).await
    }
}

impl<Txn, FE, S, T> From<DenseExpand<S>> for DenseAccess<Txn, FE, T>
where
    S: Into<DenseAccess<Txn, FE, T>>,
    T: CType,
{
    fn from(expand: DenseExpand<S>) -> Self {
        Self::Expand(Box::new(DenseExpand {
            source: expand.source.into(),
            transform: expand.transform,
        }))
    }
}

impl<S: fmt::Debug> fmt::Debug for DenseExpand<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "expand axes {:?} of {:?}",
            self.transform.expand_axes(),
            self.source,
        )
    }
}

type Combine<T> = fn(Array<T>, Array<T>) -> TCResult<Array<T>>;

#[derive(Clone)]
pub struct DenseCombine<L, R, T: CType> {
    left: L,
    right: R,
    block_op: Combine<T>,
    value_op: fn(T, T) -> T,
}

impl<L, R, T> DenseCombine<L, R, T>
where
    L: DenseInstance + fmt::Debug,
    R: DenseInstance + fmt::Debug,
    T: CType + DType,
{
    pub fn new(left: L, right: R, block_op: Combine<T>, value_op: fn(T, T) -> T) -> TCResult<Self> {
        if left.shape() == right.shape() && left.block_size() == right.block_size() {
            Ok(Self {
                left,
                right,
                block_op,
                value_op,
            })
        } else {
            Err(bad_request!(
                "cannot combine {:?} (block size {}) with {:?} (block size {})",
                left,
                left.block_size(),
                right,
                right.block_size()
            ))
        }
    }
}

impl<L, R, T> TensorInstance for DenseCombine<L, R, T>
where
    L: TensorInstance,
    R: TensorInstance,
    T: DType + CType,
{
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        debug_assert_eq!(self.left.shape(), self.right.shape());
        self.left.shape()
    }
}

#[async_trait]
impl<L, R, T> DenseInstance for DenseCombine<L, R, T>
where
    L: DenseInstance<DType = T>,
    R: DenseInstance<DType = T>,
    T: CType + DType,
{
    type Block = Array<T>;
    type DType = T;

    fn block_size(&self) -> usize {
        self.left.block_size()
    }

    async fn read_block(&self, txn_id: TxnId, block_id: u64) -> TCResult<Self::Block> {
        debug!(
            "DenseCombine::read_block {block_id} from {:?} and {:?}",
            self.left, self.right
        );

        let (left, right) = try_join!(
            self.left.read_block(txn_id, block_id),
            self.right.read_block(txn_id, block_id)
        )?;

        (self.block_op)(left.into(), right.into())
    }

    async fn read_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::Block>> {
        debug!(
            "DenseCombine::read_blocks from {:?} and {:?}",
            self.left, self.right
        );

        let op = self.block_op;
        let block_axis = block_axis_for(self.shape(), self.block_size());
        let block_shape = block_shape_for(block_axis, self.shape(), self.block_size());

        let (left, right) = try_join!(
            self.left.read_blocks(txn_id),
            self.right.read_blocks(txn_id)
        )?;

        let blocks = left.zip(right).map(move |(l, r)| {
            let (l, r) = (l?, r?);

            debug_assert!(l.shape() == block_shape, "left block has the wrong shape");
            debug_assert!(r.shape() == block_shape, "right block has the wrong shape");

            (op)(l.into(), r.into())
        });

        Ok(Box::pin(blocks))
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> TCResult<Self::DType> {
        let (left, right) = try_join!(
            self.left.read_value(txn_id, coord.clone()),
            self.right.read_value(txn_id, coord)
        )?;

        Ok((self.value_op)(left, right))
    }
}

#[async_trait]
impl<L, R, T> TensorPermitRead for DenseCombine<L, R, T>
where
    L: TensorPermitRead,
    R: TensorPermitRead,
    T: CType,
{
    async fn read_permit(
        &self,
        txn_id: TxnId,
        range: Range,
    ) -> TCResult<SmallVec<[PermitRead<Range>; 16]>> {
        // always acquire these locks in the same order, to avoid the risk of a deadlock
        let mut left = self.left.read_permit(txn_id, range.clone()).await?;
        let right = self.right.read_permit(txn_id, range).await?;
        left.extend(right);
        Ok(left)
    }
}

impl<Txn, FE, L, R, T> From<DenseCombine<L, R, T>> for DenseAccess<Txn, FE, T>
where
    L: Into<DenseAccess<Txn, FE, T>>,
    R: Into<DenseAccess<Txn, FE, T>>,
    T: CType,
{
    fn from(combine: DenseCombine<L, R, T>) -> Self {
        Self::Combine(Box::new(DenseCombine {
            left: combine.left.into(),
            right: combine.right.into(),
            block_op: combine.block_op,
            value_op: combine.value_op,
        }))
    }
}

impl<L, R, T> fmt::Debug for DenseCombine<L, R, T>
where
    L: fmt::Debug,
    R: fmt::Debug,
    T: CType + DType,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "combine {:?} with {:?}", self.left, self.right)
    }
}

#[derive(Clone)]
pub struct DenseCombineConst<S, T: CType> {
    left: S,
    right: T,
    block_op: fn(Array<T>, T) -> TCResult<Array<T>>,
    value_op: fn(T, T) -> T,
}

impl<S: TensorInstance, T: CType + DType> TensorInstance for DenseCombineConst<S, T> {
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        self.left.shape()
    }
}

#[async_trait]
impl<S, T> DenseInstance for DenseCombineConst<S, T>
where
    S: DenseInstance<DType = T>,
    T: CType + DType,
{
    type Block = Array<T>;
    type DType = T;

    fn block_size(&self) -> usize {
        self.left.block_size()
    }

    async fn read_block(&self, txn_id: TxnId, block_id: u64) -> TCResult<Self::Block> {
        let source_block = self.left.read_block(txn_id, block_id).await?;
        (self.block_op)(source_block.into(), self.right)
    }

    async fn read_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::Block>> {
        let source_blocks = self.left.read_blocks(txn_id).await?;
        let blocks = source_blocks
            .map(move |result| result.and_then(|block| (self.block_op)(block.into(), self.right)));

        Ok(Box::pin(blocks))
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> TCResult<Self::DType> {
        let left = self.left.read_value(txn_id, coord).await?;
        Ok((self.value_op)(left, self.right))
    }
}

#[async_trait]
impl<S: TensorPermitRead, T: CType> TensorPermitRead for DenseCombineConst<S, T> {
    async fn read_permit(
        &self,
        txn_id: TxnId,
        range: Range,
    ) -> TCResult<SmallVec<[PermitRead<Range>; 16]>> {
        self.left.read_permit(txn_id, range).await
    }
}

impl<Txn, FE, S, T> From<DenseCombineConst<S, T>> for DenseAccess<Txn, FE, T>
where
    S: Into<DenseAccess<Txn, FE, T>>,
    T: CType,
{
    fn from(combine: DenseCombineConst<S, T>) -> Self {
        Self::CombineConst(Box::new(DenseCombineConst {
            left: combine.left.into(),
            right: combine.right,
            block_op: combine.block_op,
            value_op: combine.value_op,
        }))
    }
}

impl<S: fmt::Debug, T: CType + DType> fmt::Debug for DenseCombineConst<S, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "combine {:?} with a constant value", self.left)
    }
}

type ArrayCmp<T> = fn(Block, Block) -> TCResult<Array<T>>;

pub struct DenseCompare<Txn, FE, T: CType> {
    left: DenseAccessCast<Txn, FE>,
    right: DenseAccessCast<Txn, FE>,
    block_op: ArrayCmp<T>,
    value_op: fn(Number, Number) -> T,
}

impl<Txn, FE, T: CType> Clone for DenseCompare<Txn, FE, T> {
    fn clone(&self) -> Self {
        Self {
            left: self.left.clone(),
            right: self.right.clone(),
            block_op: self.block_op,
            value_op: self.value_op,
        }
    }
}

impl<Txn, FE, T: CType> DenseCompare<Txn, FE, T> {
    pub fn new<L, R>(
        left: L,
        right: R,
        block_op: ArrayCmp<T>,
        value_op: fn(Number, Number) -> T,
    ) -> TCResult<Self>
    where
        L: DenseInstance + Into<DenseAccessCast<Txn, FE>> + fmt::Debug,
        R: DenseInstance + Into<DenseAccessCast<Txn, FE>> + fmt::Debug,
    {
        if left.block_size() == right.block_size() && left.shape() == right.shape() {
            Ok(Self {
                left: left.into(),
                right: right.into(),
                block_op,
                value_op,
            })
        } else {
            Err(bad_request!(
                "cannot compare {left:?} (block size {}) with {right:?} (block size {})",
                left.block_size(),
                right.block_size()
            ))
        }
    }
}

impl<Txn, FE, T> TensorInstance for DenseCompare<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CType + DType,
{
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        let left = &self.left;
        cast_dispatch!(left, this, this.shape())
    }
}

#[async_trait]
impl<Txn, FE, T> DenseInstance for DenseCompare<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node>,
    T: CType + DType + fmt::Debug,
    Number: From<T> + CastInto<T>,
{
    type Block = Array<T>;
    type DType = T;

    fn block_size(&self) -> usize {
        let left = &self.left;
        cast_dispatch!(left, this, this.block_size())
    }

    async fn read_block(&self, txn_id: TxnId, block_id: u64) -> TCResult<Self::Block> {
        let (left, right) = try_join!(
            self.left.read_block(txn_id, block_id),
            self.right.read_block(txn_id, block_id)
        )?;

        (self.block_op)(left, right)
    }

    async fn read_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::Block>> {
        let (left, right) = try_join!(
            self.left.read_blocks(txn_id),
            self.right.read_blocks(txn_id)
        )?;

        let blocks = left.zip(right).map(move |(l, r)| {
            let (l, r) = (l?, r?);
            (self.block_op)(l, r)
        });

        Ok(Box::pin(blocks))
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> TCResult<Self::DType> {
        let (left, right) = try_join!(
            self.left.read_value(txn_id, coord.clone()),
            self.right.read_value(txn_id, coord)
        )?;

        Ok((self.value_op)(left, right))
    }
}

#[async_trait]
impl<Txn: ThreadSafe, FE: ThreadSafe, T: CType> TensorPermitRead for DenseCompare<Txn, FE, T> {
    async fn read_permit(
        &self,
        txn_id: TxnId,
        range: Range,
    ) -> TCResult<SmallVec<[PermitRead<Range>; 16]>> {
        // always acquire these permits in-order to avoid the risk of a deadlock
        let mut left = self.left.read_permit(txn_id, range.clone()).await?;
        let right = self.right.read_permit(txn_id, range).await?;
        left.extend(right);
        Ok(left)
    }
}

impl<Txn, FE, T: CType> From<DenseCompare<Txn, FE, T>> for DenseAccess<Txn, FE, T> {
    fn from(compare: DenseCompare<Txn, FE, T>) -> Self {
        Self::Compare(Box::new(compare))
    }
}

impl<Txn, FE, T: CType> fmt::Debug for DenseCompare<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CType + DType,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "compare {:?} with {:?}", self.left, self.right)
    }
}

type ArrayCmpScalar<T> = fn(Block, Number) -> TCResult<Array<T>>;

pub struct DenseCompareConst<Txn, FE, T: CType> {
    left: DenseAccessCast<Txn, FE>,
    right: Number,
    block_op: ArrayCmpScalar<T>,
    value_op: fn(Number, Number) -> T,
}

impl<Txn, FE, T: CType> Clone for DenseCompareConst<Txn, FE, T> {
    fn clone(&self) -> Self {
        Self {
            left: self.left.clone(),
            right: self.right,
            block_op: self.block_op,
            value_op: self.value_op,
        }
    }
}

impl<Txn, FE, T: CType> DenseCompareConst<Txn, FE, T> {
    pub fn new<L, R>(
        left: L,
        right: R,
        block_op: ArrayCmpScalar<T>,
        value_op: fn(Number, Number) -> T,
    ) -> Self
    where
        L: Into<DenseAccessCast<Txn, FE>>,
        R: Into<Number>,
    {
        Self {
            left: left.into(),
            right: right.into(),
            block_op,
            value_op,
        }
    }
}

impl<Txn, FE, T> TensorInstance for DenseCompareConst<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CType + DType,
{
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        let left = &self.left;
        cast_dispatch!(left, this, this.shape())
    }
}

#[async_trait]
impl<Txn, FE, T> DenseInstance for DenseCompareConst<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node>,
    T: CType + DType + fmt::Debug,
{
    type Block = Array<T>;
    type DType = T;

    fn block_size(&self) -> usize {
        let left = &self.left;
        cast_dispatch!(left, this, this.block_size())
    }

    async fn read_block(&self, txn_id: TxnId, block_id: u64) -> TCResult<Self::Block> {
        self.left
            .read_block(txn_id, block_id)
            .map(|result| result.and_then(move |block| (self.block_op)(block, self.right)))
            .await
    }

    async fn read_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::Block>> {
        let left = self.left.read_blocks(txn_id).await?;
        let blocks =
            left.map(move |result| result.and_then(|block| (self.block_op)(block, self.right)));

        Ok(Box::pin(blocks))
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> TCResult<Self::DType> {
        let left = self.left.read_value(txn_id, coord).await?;
        Ok((self.value_op)(left, self.right))
    }
}

#[async_trait]
impl<Txn, FE, T> TensorPermitRead for DenseCompareConst<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CType,
{
    async fn read_permit(
        &self,
        txn_id: TxnId,
        range: Range,
    ) -> TCResult<SmallVec<[PermitRead<Range>; 16]>> {
        self.left.read_permit(txn_id, range).await
    }
}

impl<Txn, FE, T: CType> From<DenseCompareConst<Txn, FE, T>> for DenseAccess<Txn, FE, T> {
    fn from(compare: DenseCompareConst<Txn, FE, T>) -> Self {
        Self::CompareConst(Box::new(compare))
    }
}

impl<Txn, FE, T: CType> fmt::Debug for DenseCompareConst<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CType + DType,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "compare {:?} with {:?}", self.left, self.right)
    }
}

#[derive(Clone)]
pub struct DenseCond<Cond, Then, OrElse> {
    cond: Cond,
    then: Then,
    or_else: OrElse,
}

impl<Cond, Then, OrElse, T> DenseCond<Cond, Then, OrElse>
where
    Cond: DenseInstance<DType = u8> + fmt::Debug,
    Then: DenseInstance<DType = T> + fmt::Debug,
    OrElse: DenseInstance<DType = T> + fmt::Debug,
    T: CType,
{
    pub fn new(cond: Cond, then: Then, or_else: OrElse) -> TCResult<DenseCond<Cond, Then, OrElse>> {
        if cond.shape() == then.shape()
            && cond.shape() == or_else.shape()
            && then.dtype() == or_else.dtype()
            && cond.block_size() == then.block_size()
            && cond.block_size() == or_else.block_size()
        {
            Ok(Self {
                cond,
                then,
                or_else,
            })
        } else if cond.block_size() != then.block_size()
            || cond.block_size() != or_else.block_size()
        {
            Err(internal!(
                "cannot select blocks of size {cond} from blocks of size {then} and {or_else}",
                cond = cond.block_size(),
                then = then.block_size(),
                or_else = or_else.block_size()
            ))
        } else {
            Err(bad_request!(
                "cannot select conditionally from {then:?} and {or_else:?} based on {cond:?}"
            ))
        }
    }
}

impl<Cond, Then, OrElse> TensorInstance for DenseCond<Cond, Then, OrElse>
where
    Cond: TensorInstance,
    Then: TensorInstance,
    OrElse: TensorInstance,
{
    fn dtype(&self) -> NumberType {
        debug_assert_eq!(self.then.dtype(), self.or_else.dtype());
        self.then.dtype()
    }

    fn shape(&self) -> &Shape {
        debug_assert_eq!(self.cond.shape(), self.then.shape());
        debug_assert_eq!(self.cond.shape(), self.or_else.shape());
        self.cond.shape()
    }
}

#[async_trait]
impl<Cond, Then, OrElse, T> DenseInstance for DenseCond<Cond, Then, OrElse>
where
    Cond: DenseInstance<DType = u8> + Clone,
    Then: DenseInstance<DType = T> + Clone,
    OrElse: DenseInstance<DType = T> + Clone,
    T: CType + DType,
{
    type Block = ArrayOp<ha_ndarray::ops::GatherCond<Cond::Block, T, Then::Block, OrElse::Block>>;
    type DType = T;

    fn block_size(&self) -> usize {
        self.cond.block_size()
    }

    async fn read_block(&self, txn_id: TxnId, block_id: u64) -> TCResult<Self::Block> {
        trace!(
            "read block {block_id} conditionally from {:?} and {:?} based on {:?}",
            self.then,
            self.or_else,
            self.cond
        );

        let (cond, then, or_else) = try_join!(
            self.cond.read_block(txn_id, block_id),
            self.then.read_block(txn_id, block_id),
            self.or_else.read_block(txn_id, block_id),
        )?;

        cond.cond(then, or_else).map_err(TCError::from)
    }

    async fn read_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::Block>> {
        let num_blocks = div_ceil(self.size(), self.block_size() as u64);
        let blocks = futures::stream::iter(0..num_blocks)
            .map(move |block_id| {
                let this = self.clone();
                async move { this.read_block(txn_id, block_id).await }
            })
            .buffered(num_cpus::get());

        Ok(Box::pin(blocks))
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> TCResult<Self::DType> {
        let (cond, then, or_else) = try_join!(
            self.cond.read_value(txn_id, coord.clone()),
            self.then.read_value(txn_id, coord.clone()),
            self.or_else.read_value(txn_id, coord)
        )?;

        if cond != 0 {
            Ok(then)
        } else {
            Ok(or_else)
        }
    }
}

#[async_trait]
impl<Cond, Then, OrElse> TensorPermitRead for DenseCond<Cond, Then, OrElse>
where
    Cond: TensorPermitRead,
    Then: TensorPermitRead,
    OrElse: TensorPermitRead,
{
    async fn read_permit(
        &self,
        txn_id: TxnId,
        range: Range,
    ) -> TCResult<SmallVec<[PermitRead<Range>; 16]>> {
        // always aquire these permits in-order to minimize the risk of a deadlock
        let mut permit = self.cond.read_permit(txn_id, range.clone()).await?;

        let then = self.then.read_permit(txn_id, range.clone()).await?;
        permit.extend(then);

        let or_else = self.or_else.read_permit(txn_id, range.clone()).await?;
        permit.extend(or_else);

        Ok(permit)
    }
}

impl<Txn, FE, Cond, Then, OrElse, T> From<DenseCond<Cond, Then, OrElse>> for DenseAccess<Txn, FE, T>
where
    Cond: Into<DenseAccess<Txn, FE, u8>>,
    Then: Into<DenseAccess<Txn, FE, T>>,
    OrElse: Into<DenseAccess<Txn, FE, T>>,
    T: CType,
{
    fn from(cond: DenseCond<Cond, Then, OrElse>) -> Self {
        Self::Cond(Box::new(DenseCond {
            cond: cond.cond.into(),
            then: cond.then.into(),
            or_else: cond.or_else.into(),
        }))
    }
}

impl<Cond, Then, OrElse> fmt::Debug for DenseCond<Cond, Then, OrElse>
where
    Cond: fmt::Debug,
    Then: fmt::Debug,
    OrElse: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "read from {:?} or {:?} based on {:?}",
            self.then, self.or_else, self.cond
        )
    }
}

#[derive(Clone)]
pub struct DenseConst<L, T: CType> {
    left: L,
    right: T,
    block_op: fn(Array<T>, T) -> TCResult<Array<T>>,
    value_op: fn(T, T) -> T,
}

impl<L, T: CType> DenseConst<L, T> {
    pub fn new(
        left: L,
        right: T,
        block_op: fn(Array<T>, T) -> TCResult<Array<T>>,
        value_op: fn(T, T) -> T,
    ) -> Self {
        Self {
            left,
            right,
            block_op,
            value_op,
        }
    }
}

impl<L, T> TensorInstance for DenseConst<L, T>
where
    L: TensorInstance,
    T: CType + DType,
{
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        self.left.shape()
    }
}

#[async_trait]
impl<L, T> DenseInstance for DenseConst<L, T>
where
    L: DenseInstance<DType = T> + fmt::Debug,
    T: CType + DType,
{
    type Block = Array<T>;
    type DType = T;

    fn block_size(&self) -> usize {
        self.left.block_size()
    }

    async fn read_block(&self, txn_id: TxnId, block_id: u64) -> TCResult<Self::Block> {
        debug!(
            "DenseConst::read_block {block_id} from source {:?}",
            self.left
        );

        self.left
            .read_block(txn_id, block_id)
            .map(move |result| result.and_then(|block| (self.block_op)(block.into(), self.right)))
            .await
    }

    async fn read_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::Block>> {
        debug!("DenseConst::read_blocks with source {:?}", self.left);

        let left = self.left.read_blocks(txn_id).await?;
        let blocks = left.map(move |result| {
            result.and_then(|block| {
                trace!("apply const op to {block:?}");
                (self.block_op)(block.into(), self.right)
            })
        });

        Ok(Box::pin(blocks))
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> TCResult<Self::DType> {
        let left = self.left.read_value(txn_id, coord).await?;
        Ok((self.value_op)(left, self.right))
    }
}

#[async_trait]
impl<L: TensorPermitRead, T: CType> TensorPermitRead for DenseConst<L, T> {
    async fn read_permit(
        &self,
        txn_id: TxnId,
        range: Range,
    ) -> TCResult<SmallVec<[PermitRead<Range>; 16]>> {
        self.left.read_permit(txn_id, range).await
    }
}

impl<Txn, FE, L, T> From<DenseConst<L, T>> for DenseAccess<Txn, FE, T>
where
    L: Into<DenseAccess<Txn, FE, T>>,
    T: CType,
{
    fn from(combine: DenseConst<L, T>) -> Self {
        Self::Const(Box::new(DenseConst {
            left: combine.left.into(),
            right: combine.right,
            block_op: combine.block_op,
            value_op: combine.value_op,
        }))
    }
}

impl<L: fmt::Debug, T: CType + DType> fmt::Debug for DenseConst<L, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "dual constant operation on {:?}", self.left)
    }
}

#[derive(Clone)]
pub struct DenseMatMul<L, R> {
    left: L,
    right: R,
    shape: Shape,
    per_block: usize,
}

impl<L, R> DenseMatMul<L, R>
where
    L: TensorInstance + fmt::Debug,
    R: TensorInstance + fmt::Debug,
{
    pub fn new(left: L, right: R) -> TCResult<Self> {
        left.shape().validate()?;
        right.shape().validate()?;

        let per_block_left =
            (left.size() / left.shape().iter().rev().take(2).product::<u64>()) as usize;

        let per_block_right =
            (right.size() / right.shape().iter().rev().take(2).product::<u64>()) as usize;

        if per_block_left != per_block_right {
            // this is an internal error because this implementation detail should be invisible
            // but it's better to error out here than enter undefined behavior later on
            Err(internal!(
                "cannot matrix multiply {left:?} with {right:?} due to different block sizes"
            ))
        } else if left.ndim() != right.ndim() {
            Err(bad_request!("cannot matrix multiply {left:?} with {right:?} since they have different dimensions"))
        } else if left.shape().last() != right.shape().iter().nth_back(1) {
            Err(bad_request!(
                "cannot matrix multiply {left:?} with {right:?} due to non-matching dimensions"
            ))
        } else {
            let mut shape = SmallVec::<[u64; 8]>::with_capacity(left.ndim());
            shape.extend_from_slice(&left.shape()[..left.ndim() - 1]);
            shape.push(right.shape()[right.ndim() - 1]);

            Ok(Self {
                left,
                right,
                shape: shape.into(),
                per_block: per_block_left,
            })
        }
    }

    #[inline]
    fn block_op<T: CType>(left: Array<T>, right: Array<T>) -> TCResult<Array<T>> {
        left.matmul(right).map(Array::from).map_err(TCError::from)
    }
}

impl<L, R> TensorInstance for DenseMatMul<L, R>
where
    L: TensorInstance,
    R: TensorInstance,
{
    fn dtype(&self) -> NumberType {
        self.left.dtype()
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }
}

#[async_trait]
impl<L, R, T> DenseInstance for DenseMatMul<L, R>
where
    L: DenseInstance<DType = T> + Clone,
    R: DenseInstance<DType = T> + Clone,
    T: CType + DType,
{
    type Block = Array<T>;
    type DType = T;

    fn block_size(&self) -> usize {
        self.per_block * self.shape.iter().rev().take(2).product::<u64>() as usize
    }

    async fn read_block(&self, txn_id: TxnId, block_id: u64) -> TCResult<Self::Block> {
        let (left, right) = try_join!(
            self.left.read_block(txn_id, block_id),
            self.right.read_block(txn_id, block_id),
        )?;

        Self::block_op(left.into(), right.into())
    }

    async fn read_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::Block>> {
        let num_blocks = self.size() / self.block_size() as u64;

        let blocks = futures::stream::iter(0..num_blocks)
            .map(move |block_id| {
                let this = self.clone();
                async move { this.read_block(txn_id, block_id).await }
            })
            .buffered(num_cpus::get());

        Ok(Box::pin(blocks))
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> TCResult<Self::DType> {
        self.shape.validate_coord(&coord)?;

        let mut right_range = SmallVec::<[AxisRange; 8]>::with_capacity(coord.len());
        right_range.extend(
            coord
                .iter()
                .copied()
                .map(AxisRange::At)
                .take(self.ndim() - 2),
        );
        right_range.push(AxisRange::In(0..self.shape[self.ndim() - 2], 1));
        right_range.push(AxisRange::At(coord[self.ndim() - 1]));
        let right_range = Range::from(right_range);

        let left_range = coord
            .into_iter()
            .take(self.ndim() - 1)
            .map(AxisRange::At)
            .collect::<Range>();

        let coords = left_range.affected().zip(right_range.affected());

        let mut sum = T::zero();
        for (left_coord, right_coord) in coords {
            let (left, right) = try_join!(
                self.left.read_value(txn_id, left_coord.into()),
                self.right.read_value(txn_id, right_coord.into())
            )?;

            sum = sum + (left * right);
        }

        Ok(sum)
    }
}

#[async_trait]
impl<L, R> TensorPermitRead for DenseMatMul<L, R>
where
    L: TensorInstance + TensorPermitRead,
    R: TensorInstance + TensorPermitRead,
{
    async fn read_permit(
        &self,
        txn_id: TxnId,
        range: Range,
    ) -> TCResult<SmallVec<[PermitRead<Range>; 16]>> {
        self.shape.validate_range(&range)?;

        let left_range = range.iter().take(self.left.ndim() - 1).cloned().collect();

        let mut right_range = range;
        let elided_right = self.right.ndim() - 2;
        if right_range.len() > elided_right {
            right_range[elided_right] = AxisRange::all(self.right.shape()[elided_right]);
        }

        // always acquire these permits in-order to avoid the risk of a deadlock
        let mut left = self.left.read_permit(txn_id, left_range).await?;
        let right = self.right.read_permit(txn_id, right_range).await?;
        left.extend(right);
        Ok(left)
    }
}

impl<Txn, FE, L, R, T> From<DenseMatMul<L, R>> for DenseAccess<Txn, FE, T>
where
    T: CType,
    L: Into<DenseAccess<Txn, FE, T>>,
    R: Into<DenseAccess<Txn, FE, T>>,
{
    fn from(matmul: DenseMatMul<L, R>) -> Self {
        Self::MatMul(Box::new(DenseMatMul {
            left: matmul.left.into(),
            right: matmul.right.into(),
            shape: matmul.shape,
            per_block: matmul.per_block,
        }))
    }
}

impl<L: fmt::Debug, R: fmt::Debug> fmt::Debug for DenseMatMul<L, R> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "matrix multiply {:?} with {:?}", self.left, self.right)
    }
}

#[derive(Clone)]
pub struct DenseReduce<S, T: CType> {
    source: S,
    transform: Reduce,
    block_map: ArrayBuf<StackVec<u64>>,
    map_axes: Axes,
    block_axes: Axes,
    id: T,
    reduce_all: fn(Array<T>, T) -> TCResult<T>,
    reduce_blocks: Combine<T>,
    reduce_op: fn(Array<T>, &[usize], bool) -> TCResult<Array<T>>,
}

impl<S: DenseInstance> DenseReduce<S, S::DType> {
    fn new(
        source: S,
        mut axes: Axes,
        keepdims: bool,
        id: S::DType,
        reduce_all: fn(Array<S::DType>, S::DType) -> TCResult<S::DType>,
        reduce_blocks: Combine<S::DType>,
        reduce_op: fn(Array<S::DType>, &[usize], bool) -> TCResult<Array<S::DType>>,
    ) -> TCResult<Self> {
        axes.sort();
        axes.dedup();

        let num_blocks = div_ceil(source.size(), source.block_size() as u64);
        let block_axis = block_axis_for(source.shape(), source.block_size());
        let block_shape = block_shape_for(block_axis, source.shape(), source.block_size());
        debug_assert_eq!(source.shape()[block_axis] % block_shape[0] as u64, 0);

        let map_axes = axes.iter().copied().filter(|x| *x <= block_axis).collect();
        let block_axes = axes
            .iter()
            .copied()
            .filter(|x| *x >= block_axis)
            .map(|x| x - block_axis)
            .collect();

        let transform = Reduce::new(source.shape().clone(), axes, keepdims)?;

        let mut block_map_shape = Vec::with_capacity(source.ndim());
        block_map_shape.extend(
            source.shape()[..block_axis]
                .iter()
                .copied()
                .map(|dim| dim as usize),
        );
        block_map_shape.push(source.shape()[block_axis] as usize / block_shape[0]);

        let block_map = (0..num_blocks).into_iter().collect();
        let block_map = ArrayBuf::new(block_map_shape, block_map)?;

        Ok(Self {
            source,
            transform,
            block_map,
            map_axes,
            block_axes,
            id,
            reduce_all,
            reduce_blocks,
            reduce_op,
        })
    }

    pub fn max(source: S, axes: Axes, keepdims: bool) -> TCResult<Self>
    where
        Number: From<S::DType>,
    {
        Self::new(
            source,
            axes,
            keepdims,
            S::DType::min(),
            |block, max| {
                let block_max = block.max_all()?;
                let max = match NumberCollator::default().cmp(&max.into(), &block_max.into()) {
                    Ordering::Less => block_max,
                    Ordering::Equal | Ordering::Greater => max,
                };

                Ok(max)
            },
            |l, r| {
                l.ge(r)?
                    .cond(l, r)
                    .map(Array::from)
                    .map_err(TCError::from)
            },
            |block, axes, keepdims| {
                block
                    .max(axes.to_vec(), keepdims)
                    .map(Array::from)
                    .map_err(TCError::from)
            },
        )
    }

    pub fn min(source: S, axes: Axes, keepdims: bool) -> TCResult<Self>
    where
        Number: From<S::DType>,
    {
        Self::new(
            source,
            axes,
            keepdims,
            S::DType::max(),
            |block, min| {
                let block_min = block.min_all()?;
                let min = match NumberCollator::default().cmp(&min.into(), &block_min.into()) {
                    Ordering::Less | Ordering::Equal => min,
                    Ordering::Greater => block_min,
                };

                Ok(min)
            },
            |l, r| {
                l.le(r)?
                    .cond(l, r)
                    .map(Array::from)
                    .map_err(TCError::from)
            },
            |block, axes, keepdims| {
                block
                    .min(axes.to_vec(), keepdims)
                    .map(Array::from)
                    .map_err(TCError::from)
            },
        )
    }

    pub fn sum(source: S, axes: Axes, keepdims: bool) -> TCResult<Self> {
        Self::new(
            source,
            axes,
            keepdims,
            S::DType::zero(),
            |block, sum| {
                block
                    .sum_all()
                    .map(|block_sum| block_sum + sum)
                    .map_err(TCError::from)
            },
            |l, r| l.add(r).map(Array::from).map_err(TCError::from),
            |block, axes, keepdims| {
                block
                    .sum(axes.to_vec(), keepdims)
                    .map(Array::from)
                    .map_err(TCError::from)
            },
        )
    }

    pub fn product(source: S, axes: Axes, keepdims: bool) -> TCResult<Self> {
        Self::new(
            source,
            axes,
            keepdims,
            S::DType::one(),
            |block, product| {
                let block_product = block.product_all()?;
                Ok(block_product * product)
            },
            |l, r| l.mul(r).map(Array::from).map_err(TCError::from),
            |block, axes, keepdims| {
                block
                    .product(axes.to_vec(), keepdims)
                    .map(Array::from)
                    .map_err(TCError::from)
            },
        )
    }
}

impl<S: TensorInstance, T: CType> TensorInstance for DenseReduce<S, T> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
        self.transform.shape()
    }
}

#[async_trait]
impl<S, T> DenseInstance for DenseReduce<S, T>
where
    S: DenseInstance<Block = Array<T>, DType = T> + Clone,
    T: CType + DType,
{
    type Block = Array<T>;
    type DType = T;

    fn block_size(&self) -> usize {
        let block_axis = block_axis_for(self.source.shape(), self.source.block_size());
        let block_shape =
            block_shape_for(block_axis, self.source.shape(), self.source.block_size());

        block_shape
            .iter()
            .enumerate()
            .filter_map(|(x, dim)| {
                if self.block_axes.contains(&x) {
                    None
                } else {
                    Some(dim)
                }
            })
            .product()
    }

    async fn read_block(&self, txn_id: TxnId, block_id: u64) -> TCResult<Self::Block> {
        let source_blocks_per_block = self
            .map_axes
            .iter()
            .copied()
            .map(|x| self.block_map.shape()[x])
            .product::<usize>();

        let source_block_id = block_id * source_blocks_per_block as u64;
        let map_strides = ha_ndarray::strides_for(self.block_map.shape(), self.block_map.ndim());
        let map_coord = coord_of(
            source_block_id as usize,
            &map_strides,
            self.block_map.shape(),
            0,
        );

        let source_blocks = if !self.map_axes.is_empty() {
            let mut map_slice = map_coord
                .into_iter()
                .map(|i| ha_ndarray::AxisRange::At(i))
                .collect::<Vec<_>>();

            for x in self.map_axes.iter().copied() {
                map_slice[x] = ha_ndarray::AxisRange::In(0, self.block_map.shape()[x], 1);
            }

            let queue = autoqueue(&self.block_map)?;
            let block_map_slice = self.block_map.clone().slice(map_slice)?;
            block_map_slice.read(&queue)?.to_slice()?.into_vec()
        } else {
            let block_id = self.block_map.read_value(&map_coord)?;
            vec![block_id]
        };

        debug_assert_eq!(source_blocks[0], source_block_id);

        let block = self
            .source
            .read_block(txn_id, source_block_id)
            .map(|result| {
                result.and_then(|block| {
                    debug_assert!(self.block_axes.iter().copied().all(|x| x < block.ndim()));
                    (self.reduce_op)(block, &self.block_axes, self.transform.keepdims())
                })
            })
            .await?;

        futures::stream::iter(source_blocks.into_iter().skip(1))
            .map(|source_block_id| {
                self.source
                    .read_block(txn_id, source_block_id)
                    .map(|result| {
                        result.and_then(|block| {
                            debug_assert!(self
                                .block_axes
                                .iter()
                                .copied()
                                .all(|x| x < block.ndim()));
                            (self.reduce_op)(block, &self.block_axes, self.transform.keepdims())
                        })
                    })
            })
            .buffer_unordered(num_cpus::get())
            .try_fold(block, |block, source_block| async move {
                (self.reduce_blocks)(block, source_block)
            })
            .await
    }

    // TODO: optimize
    async fn read_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::Block>> {
        let num_blocks = div_ceil(self.size(), self.block_size() as u64);
        let blocks = futures::stream::iter(0..num_blocks)
            .map(move |block_id| {
                let this = self.clone();
                async move { this.read_block(txn_id, block_id).await }
            })
            .buffered(num_cpus::get());

        Ok(Box::pin(blocks))
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> TCResult<Self::DType> {
        self.shape().validate_coord(&coord)?;
        let range = self.transform.invert_coord(&coord);
        let slice = DenseSlice::new(self.source.clone(), range)?;
        let source_blocks = slice.read_blocks(txn_id).await?;
        source_blocks
            .try_fold(self.id, |reduced, block| async move {
                (self.reduce_all)(block, reduced)
            })
            .await
    }
}

#[async_trait]
impl<S: TensorPermitRead, T: CType> TensorPermitRead for DenseReduce<S, T> {
    async fn read_permit(
        &self,
        txn_id: TxnId,
        range: Range,
    ) -> TCResult<SmallVec<[PermitRead<Range>; 16]>> {
        self.transform.shape().validate_range(&range)?;
        let range = self.transform.invert_range(range);
        self.source.read_permit(txn_id, range).await
    }
}

impl<Txn, FE, S, T> From<DenseReduce<S, T>> for DenseAccess<Txn, FE, T>
where
    S: Into<DenseAccess<Txn, FE, T>>,
    T: CType,
{
    fn from(reduce: DenseReduce<S, T>) -> Self {
        Self::Reduce(Box::new(DenseReduce {
            source: reduce.source.into(),
            block_map: reduce.block_map,
            block_axes: reduce.block_axes,
            map_axes: reduce.map_axes,
            id: reduce.id,
            transform: reduce.transform,
            reduce_all: reduce.reduce_all,
            reduce_blocks: reduce.reduce_blocks,
            reduce_op: reduce.reduce_op,
        }))
    }
}

impl<S: fmt::Debug, T: CType + DType> fmt::Debug for DenseReduce<S, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "reduce axes {:?} of {:?}",
            self.transform.reduce_axes(),
            self.source
        )
    }
}

#[derive(Clone)]
pub struct DenseReshape<S> {
    source: S,
    transform: Reshape,
}

impl<S: DenseInstance> DenseReshape<S> {
    pub fn new(source: S, shape: Shape) -> TCResult<Self> {
        Reshape::new(source.shape().clone(), shape).map(|transform| Self { source, transform })
    }
}

impl<S: TensorInstance> TensorInstance for DenseReshape<S> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
        self.transform.shape()
    }
}

#[async_trait]
impl<S: DenseInstance> DenseInstance for DenseReshape<S>
where
    S::Block: NDArrayTransform,
    <S::Block as NDArrayTransform>::Reshape:
        NDArrayRead<DType = S::DType> + NDArrayTransform + Into<Array<S::DType>>,
{
    type Block = <S::Block as NDArrayTransform>::Reshape;
    type DType = S::DType;

    fn block_size(&self) -> usize {
        self.source.block_size()
    }

    async fn read_block(&self, txn_id: TxnId, block_id: u64) -> TCResult<Self::Block> {
        let block_axis = block_axis_for(self.shape(), self.block_size());
        let mut block_shape = block_shape_for(block_axis, self.shape(), self.block_size());

        let block = self.source.read_block(txn_id, block_id).await?;

        if block.size() < self.block_size() {
            // this must be the trailing block
            let axis_dim = self.block_size() / block_shape.iter().skip(1).product::<usize>();
            block_shape[0] = axis_dim;
        }

        block.reshape(block_shape).map_err(TCError::from)
    }

    async fn read_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::Block>> {
        let block_size = self.block_size();
        let block_axis = block_axis_for(self.shape(), block_size);
        let block_shape = block_shape_for(block_axis, self.shape(), block_size);

        let source_blocks = self.source.read_blocks(txn_id).await?;
        let blocks = source_blocks.map(move |result| {
            let block = result?;
            let mut block_shape = block_shape.to_vec();

            if block.size() < block_size {
                // this must be the trailing block
                let axis_dim = block_size / block_shape.iter().skip(1).product::<usize>();
                block_shape[0] = axis_dim;
            }

            block.reshape(block_shape).map_err(TCError::from)
        });

        Ok(Box::pin(blocks))
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> TCResult<Self::DType> {
        self.shape().validate_coord(&coord)?;
        let coord = self.transform.invert_coord(coord);
        self.source.read_value(txn_id, coord).await
    }
}

#[async_trait]
impl<S: TensorPermitRead + fmt::Debug> TensorPermitRead for DenseReshape<S> {
    async fn read_permit(
        &self,
        txn_id: TxnId,
        range: Range,
    ) -> TCResult<SmallVec<[PermitRead<Range>; 16]>> {
        if self.transform.shape().is_covered_by(&range) {
            self.read_permit(txn_id, Range::default()).await
        } else {
            Err(bad_request!(
                "cannot lock range {:?} of {:?} for reading (consider making a copy first)",
                range,
                self
            ))
        }
    }
}

impl<Txn, FE, T, S> From<DenseReshape<S>> for DenseAccess<Txn, FE, T>
where
    T: CType,
    S: Into<DenseAccess<Txn, FE, T>>,
{
    fn from(reshape: DenseReshape<S>) -> Self {
        Self::Reshape(Box::new(DenseReshape {
            source: reshape.source.into(),
            transform: reshape.transform,
        }))
    }
}

impl<S: fmt::Debug> fmt::Debug for DenseReshape<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "reshape of {:?} into {:?}",
            self.source,
            self.transform.shape()
        )
    }
}

#[derive(Clone)]
pub struct DenseResizeBlocks<S> {
    source: S,
    block_size: usize,
}

impl<S> DenseResizeBlocks<S> {
    pub fn new(source: S, block_size: usize) -> Self {
        Self { source, block_size }
    }
}

impl<S: TensorInstance> TensorInstance for DenseResizeBlocks<S> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
        self.source.shape()
    }
}

#[async_trait]
impl<S: DenseInstance> DenseInstance for DenseResizeBlocks<S> {
    type Block = ArrayBuf<Buffer<S::DType>>;
    type DType = S::DType;

    fn block_size(&self) -> usize {
        self.block_size
    }

    async fn read_block(&self, txn_id: TxnId, block_id: u64) -> TCResult<Self::Block> {
        let block_axis = block_axis_for(self.shape(), self.block_size);
        let mut block_shape = block_shape_for(block_axis, self.shape(), self.block_size);

        let block_size = self.block_size as u64;
        let source_block_size = self.source.block_size() as u64;

        let start = block_id * block_size;
        let stop = start + block_size;

        let source_block_id_start = start / source_block_size;
        let source_block_id_stop = div_ceil(stop, source_block_size);
        let num_source_blocks = source_block_id_stop - source_block_id_start;
        assert_ne!(num_source_blocks, 0);

        let mut source_buffers = futures::stream::iter(source_block_id_start..source_block_id_stop)
            .map(|block_id| self.source.read_block(txn_id, block_id))
            .buffered(num_cpus::get())
            .map_ok(|block| async move {
                let queue = autoqueue(&block)?;
                let buffer = block.read(&queue)?;

                buffer
                    .to_slice()
                    .map(|buffer| buffer.into_vec())
                    .map_err(TCError::from)
            })
            .try_buffered(num_cpus::get());

        let mut buffer = Vec::with_capacity(self.block_size);

        {
            let source_buffer = source_buffers.try_next().await?;
            let source_buffer = source_buffer.expect("source buffer");

            let start = (start % source_block_size) as usize;
            let stop = Ord::min(start + self.block_size, source_buffer.len());
            buffer.extend_from_slice(&source_buffer[start..stop]);
        }

        if num_source_blocks > 2 {
            for _ in 0..(num_source_blocks - 2) {
                let source_buffer = source_buffers.try_next().await?;
                let source_buffer = source_buffer.expect("source buffer");
                buffer.extend(source_buffer);
            }
        }

        if let Some(source_buffer) = source_buffers.try_next().await? {
            let stop = (stop - ((source_block_id_stop - 1) * source_block_size)) as usize;
            buffer.extend_from_slice(&source_buffer[0..stop]);
        }

        if stop > self.size() {
            let trailing_size = block_shape.iter().skip(1).product::<usize>();
            debug_assert_eq!(buffer.len() % trailing_size, 0);
            block_shape[0] = buffer.len() / trailing_size;
        }

        ArrayBuf::new(block_shape, buffer).map_err(TCError::from)
    }

    async fn read_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::Block>> {
        let block_axis = block_axis_for(self.shape(), self.block_size);
        let block_shape = block_shape_for(block_axis, self.shape(), self.block_size);
        let source_blocks = self.source.read_blocks(txn_id).await?;
        let blocks = BlockResize::new(source_blocks, block_shape)?;
        Ok(Box::pin(blocks))
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> TCResult<Self::DType> {
        self.source.read_value(txn_id, coord).await
    }
}

#[async_trait]
impl<S: TensorPermitRead> TensorPermitRead for DenseResizeBlocks<S> {
    async fn read_permit(
        &self,
        txn_id: TxnId,
        range: Range,
    ) -> TCResult<SmallVec<[PermitRead<Range>; 16]>> {
        self.source.read_permit(txn_id, range).await
    }
}

impl<Txn, FE, T, S> From<DenseResizeBlocks<S>> for DenseAccess<Txn, FE, T>
where
    T: CType,
    S: Into<DenseAccess<Txn, FE, T>>,
{
    fn from(resize: DenseResizeBlocks<S>) -> Self {
        Self::ResizeBlocks(Box::new(DenseResizeBlocks {
            source: resize.source.into(),
            block_size: resize.block_size,
        }))
    }
}

impl<S: fmt::Debug> fmt::Debug for DenseResizeBlocks<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "resize blocks of {:?}", self.source)
    }
}

#[derive(Clone)]
pub struct DenseSlice<S> {
    source: S,
    transform: Slice,
    block_map: ArrayBuf<StackVec<u64>>,
    block_size: usize,
}

impl<S: DenseInstance> DenseSlice<S> {
    pub fn new(source: S, range: Range) -> TCResult<Self> {
        debug!("construct a slice {range:?} of {source:?}");

        let transform = Slice::new(source.shape().clone(), range)?;

        let block_axis = block_axis_for(source.shape(), source.block_size());
        let block_shape = block_shape_for(block_axis, source.shape(), source.block_size());
        let num_blocks = div_ceil(
            source.size() as u64,
            block_shape.iter().product::<usize>() as u64,
        );

        let size = transform.shape().iter().product::<u64>();
        let block_size = size / num_blocks;
        debug_assert_eq!(block_size * num_blocks, size);

        let block_map = Self::block_map(block_axis, num_blocks, &transform, block_shape)?;

        Ok(Self {
            source,
            transform,
            block_map,
            block_size: block_size as usize,
        })
    }

    fn block_map(
        block_axis: usize,
        num_blocks: u64,
        transform: &Slice,
        block_shape: BlockShape,
    ) -> TCResult<ArrayBuf<StackVec<u64>>> {
        debug!(
            "construct block map for slice {:?} of {:?} with block shape {block_shape:?}",
            transform.range(),
            transform.source_shape(),
        );

        let block_map = block_map_for(num_blocks, transform.source_shape(), &block_shape)?;

        let mut block_map_bounds = Vec::with_capacity(block_axis + 1);
        for axis_range in transform.range().iter().take(block_axis).cloned() {
            let bound = ha_ndarray::AxisRange::try_from(axis_range)?;
            block_map_bounds.push(bound);
        }

        if transform.range().len() > block_axis {
            let stride = block_shape[0];
            let bound = match &transform.range()[block_axis] {
                AxisRange::At(i) if block_map_bounds.iter().all(|b| b.is_index()) => {
                    let i = (i / stride as u64) as usize;
                    ha_ndarray::AxisRange::In(i, i + 1, 1)
                }
                AxisRange::At(i) => {
                    let i = (i / stride as u64) as usize;
                    ha_ndarray::AxisRange::At(i)
                }
                AxisRange::In(axis_range, _step) => {
                    let start = (axis_range.start / stride as u64) as usize;
                    let stop = div_ceil(axis_range.end, stride as u64) as usize;
                    ha_ndarray::AxisRange::In(start, stop, 1)
                }
                AxisRange::Of(indices) => {
                    let indices = indices
                        .iter()
                        .copied()
                        .map(|i| (i / stride as u64) as usize)
                        .collect::<Vec<usize>>();

                    ha_ndarray::AxisRange::Of(indices)
                }
            };

            block_map_bounds.push(bound);
        }

        debug!("slice block map is range {block_map_bounds:?} of {block_map:?}");

        let block_map = if block_map_bounds.is_empty() {
            block_map
        } else {
            let block_map = block_map.slice(block_map_bounds)?;
            ArrayBuf::<StackVec<u64>>::copy(&block_map)?
        };

        Ok(block_map)
    }

    #[inline]
    fn block_bounds_inner(
        transform: &Slice,
        block_axis: usize,
        block_shape: &BlockShape,
        num_blocks: u64,
        source_block_id: u64,
    ) -> TCResult<Vec<ha_ndarray::AxisRange>> {
        let mut block_bounds = Vec::with_capacity(transform.source_shape().len());
        for bound in transform.range().iter().skip(block_axis).cloned() {
            block_bounds.push(bound.try_into()?);
        }

        if block_bounds.len() < block_shape.len()
            || (block_shape[0] as u64) < transform.shape()[block_axis]
        {
            let axis_bound = transform.range()[block_axis].clone();
            let block_axis_bound = ha_ndarray::AxisRange::try_from(axis_bound)?;
            trace!("block axis bound is {block_axis_bound:?}");
            let local_bound = match block_axis_bound {
                ha_ndarray::AxisRange::At(i) => ha_ndarray::AxisRange::At(i),
                ha_ndarray::AxisRange::In(start, stop, step) => {
                    let stride = block_shape[0];

                    if source_block_id == 0 {
                        ha_ndarray::AxisRange::In(start, stride, step)
                    } else if source_block_id == num_blocks - 1 {
                        ha_ndarray::AxisRange::In(stop - (stop % stride), stop, step)
                    } else {
                        let start = source_block_id as usize * stride;
                        ha_ndarray::AxisRange::In(start, start + stride, step)
                    }
                }
                ha_ndarray::AxisRange::Of(indices) => {
                    if source_block_id < indices.len() as u64 {
                        let i = indices[source_block_id as usize] as usize;
                        ha_ndarray::AxisRange::At(i)
                    } else {
                        return Err(bad_request!("block id {} is out of range", source_block_id));
                    }
                }
            };

            trace!(
                "local (block axis) bound for source block {source_block_id} is {local_bound:?}"
            );

            block_bounds.insert(0, local_bound);
        }

        debug_assert_eq!(
            block_shape.len(),
            block_bounds.iter().filter(|bound| bound.size() > 0).count()
        );

        Ok(block_bounds)
    }

    #[inline]
    fn block_bounds(&self, block_id: u64) -> TCResult<(u64, Vec<ha_ndarray::AxisRange>)> {
        let source_block_id = source_block_id_for(&self.block_map, block_id)?;
        let num_blocks = self.block_map.size() as u64;
        let block_axis = block_axis_for(self.shape(), self.block_size);
        let block_shape = block_shape_for(block_axis, self.shape(), self.block_size);
        let block_bounds = Self::block_bounds_inner(
            &self.transform,
            block_axis,
            &block_shape,
            num_blocks,
            source_block_id,
        )?;

        Ok((source_block_id, block_bounds))
    }
}

impl<S: DenseInstance + Clone> DenseSlice<S> {
    async fn block_stream<Get, Fut, Block>(
        self,
        get_block: Get,
    ) -> TCResult<impl Stream<Item = TCResult<Block::Slice>>>
    where
        Get: Fn(S, u64) -> Fut + Copy,
        Fut: Future<Output = TCResult<Block>>,
        Block: NDArrayTransform,
    {
        let block_axis = block_axis_for(self.shape(), self.block_size);
        let block_shape = block_shape_for(block_axis, self.shape(), self.block_size);
        let num_blocks = self.block_map.size() as u64;

        let source = self.source;
        let transform = self.transform;

        let blocks = futures::stream::iter(self.block_map.into_inner())
            .map(move |source_block_id| {
                get_block(source.clone(), source_block_id)
                    .map_ok(move |block| (source_block_id, block))
            })
            .buffered(num_cpus::get())
            .map(move |result| {
                let (source_block_id, source_block) = result?;
                let block_bounds = Self::block_bounds_inner(
                    &transform,
                    block_axis,
                    &block_shape,
                    num_blocks,
                    source_block_id,
                )?;

                assert_eq!(source_block.ndim(), block_bounds.len());
                debug_assert_eq!(
                    block_bounds
                        .iter()
                        .map(|bound| bound.size())
                        .filter(|dim| dim > &0)
                        .collect::<BlockShape>(),
                    block_shape
                );

                trace!("slice {block_bounds:?} from {source_block:?}");
                source_block.slice(block_bounds).map_err(TCError::from)
            });

        Ok(Box::pin(blocks))
    }
}

impl<S: TensorInstance> TensorInstance for DenseSlice<S> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
        self.transform.shape()
    }
}

#[async_trait]
impl<S: DenseInstance + Clone> DenseInstance for DenseSlice<S>
where
    S::Block: NDArrayTransform,
    <S::Block as NDArrayTransform>::Slice:
        NDArrayRead<DType = S::DType> + NDArrayTransform + Into<Array<S::DType>>,
{
    type Block = <S::Block as NDArrayTransform>::Slice;
    type DType = S::DType;

    fn block_size(&self) -> usize {
        self.block_size
    }

    async fn read_block(&self, txn_id: TxnId, block_id: u64) -> TCResult<Self::Block> {
        debug!(
            "DenseSlice::read_block {block_id} from source {:?}",
            self.source
        );

        let block_axis = block_axis_for(self.shape(), self.block_size);
        let block_shape = block_shape_for(block_axis, self.shape(), self.block_size);

        let (source_block_id, block_bounds) = self.block_bounds(block_id)?;
        let source_block = self.source.read_block(txn_id, source_block_id).await?;

        assert_eq!(source_block.ndim(), block_bounds.len());
        debug_assert_eq!(
            block_bounds
                .iter()
                .map(|bound| bound.size())
                .filter(|dim| dim > &0)
                .collect::<BlockShape>(),
            block_shape
        );

        trace!("slice {block_bounds:?} from {source_block:?}");
        source_block.slice(block_bounds).map_err(TCError::from)
    }

    async fn read_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::Block>> {
        debug!("DenseSlice::read_blocks from source {:?}", self.source);

        let blocks = self
            .block_stream(move |source, block_id| async move {
                source.read_block(txn_id, block_id).await
            })
            .await?;

        Ok(Box::pin(blocks))
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> TCResult<Self::DType> {
        self.shape().validate_coord(&coord)?;
        let coord = self.transform.invert_coord(coord);
        self.source.read_value(txn_id, coord).await
    }
}

#[async_trait]
impl<S: TensorPermitRead> TensorPermitRead for DenseSlice<S> {
    async fn read_permit(
        &self,
        txn_id: TxnId,
        range: Range,
    ) -> TCResult<SmallVec<[PermitRead<Range>; 16]>> {
        self.transform.shape().validate_range(&range)?;
        let range = self.transform.invert_range(range);
        self.source.read_permit(txn_id, range).await
    }
}

#[async_trait]
impl<S: TensorPermitWrite> TensorPermitWrite for DenseSlice<S> {
    async fn write_permit(&self, txn_id: TxnId, range: Range) -> TCResult<PermitWrite<Range>> {
        self.transform.shape().validate_range(&range)?;
        let range = self.transform.invert_range(range);
        self.source.write_permit(txn_id, range).await
    }
}

#[async_trait]
impl<'a, S: DenseWrite + Clone> DenseWrite for DenseSlice<S>
where
    S::Block: NDArrayTransform,
    <S::Block as NDArrayTransform>::Slice:
        NDArrayRead<DType = S::DType> + NDArrayTransform + Into<Array<S::DType>>,
    S::BlockWrite: NDArrayTransform,
    <S::BlockWrite as NDArrayTransform>::Slice:
        NDArrayRead<DType = S::DType> + NDArrayTransform + NDArrayWrite + Into<Array<S::DType>>,
{
    type BlockWrite = <S::BlockWrite as NDArrayTransform>::Slice;

    async fn write_block(&self, txn_id: TxnId, block_id: u64) -> TCResult<Self::BlockWrite> {
        let (source_block_id, block_bounds) = self.block_bounds(block_id)?;
        let source_block = self.source.write_block(txn_id, source_block_id).await?;
        source_block.slice(block_bounds).map_err(TCError::from)
    }

    async fn write_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::BlockWrite>> {
        let blocks = self
            .block_stream(move |source, block_id| async move {
                source.write_block(txn_id, block_id).await
            })
            .await?;

        Ok(Box::pin(blocks))
    }
}

#[async_trait]
impl<'a, S: DenseWrite + DenseWriteLock<'a> + Clone> DenseWriteLock<'a> for DenseSlice<S>
where
    S::Block: NDArrayTransform,
    <S::Block as NDArrayTransform>::Slice:
        NDArrayRead<DType = S::DType> + NDArrayTransform + Into<Array<S::DType>>,
    S::BlockWrite: NDArrayTransform,
    <S::BlockWrite as NDArrayTransform>::Slice:
        NDArrayRead<DType = S::DType> + NDArrayTransform + NDArrayWrite + Into<Array<S::DType>>,
{
    type WriteGuard = DenseSliceWriteGuard<'a, S>;

    async fn write(&'a self) -> Self::WriteGuard {
        DenseSliceWriteGuard { dest: self }
    }
}

impl<Txn, FE, S, T> From<DenseSlice<S>> for DenseAccess<Txn, FE, T>
where
    T: CType,
    S: Into<DenseAccess<Txn, FE, T>>,
{
    fn from(slice: DenseSlice<S>) -> Self {
        Self::Slice(Box::new(DenseSlice {
            source: slice.source.into(),
            transform: slice.transform,
            block_map: slice.block_map,
            block_size: slice.block_size,
        }))
    }
}

impl<S: fmt::Debug> fmt::Debug for DenseSlice<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "slice {:?} from {:?}",
            self.transform.range(),
            self.source
        )
    }
}

pub struct DenseSliceWriteGuard<'a, S> {
    dest: &'a DenseSlice<S>,
}

#[async_trait]
impl<'a, S> DenseWriteGuard<S::DType> for DenseSliceWriteGuard<'a, S>
where
    S: DenseWrite + DenseWriteLock<'a> + Clone,
    S::Block: NDArrayTransform,
    <S::Block as NDArrayTransform>::Slice:
        NDArrayRead<DType = S::DType> + NDArrayTransform + Into<Array<S::DType>>,
    S::BlockWrite: NDArrayTransform,
    <S::BlockWrite as NDArrayTransform>::Slice:
        NDArrayRead<DType = S::DType> + NDArrayTransform + NDArrayWrite + Into<Array<S::DType>>,
{
    async fn overwrite<O: DenseInstance<DType = S::DType>>(
        &self,
        txn_id: TxnId,
        other: O,
    ) -> TCResult<()> {
        let block_axis = block_axis_for(self.dest.shape(), self.dest.block_size);
        let block_shape = block_shape_for(block_axis, self.dest.shape(), self.dest.block_size);

        let dest = self.dest.clone().write_blocks(txn_id).await?;
        let source = other.read_blocks(txn_id).await?;
        let source = BlockResize::new(source, block_shape)?;

        dest.zip(source)
            .map(|(dest, source)| {
                let mut dest = dest?;
                let source = source?;
                dest.write(&source).map_err(TCError::from)
            })
            .try_fold((), |(), _| futures::future::ready(Ok(())))
            .await
    }

    async fn overwrite_value(&self, txn_id: TxnId, value: S::DType) -> TCResult<()> {
        let dest = self.dest.clone().write_blocks(txn_id).await?;
        dest.map_ok(|mut block| block.write_value(value))
            .try_fold((), |(), _| futures::future::ready(Ok(())))
            .await
    }

    async fn write_value(&self, txn_id: TxnId, coord: Coord, value: S::DType) -> TCResult<()> {
        let source_coord = self.dest.transform.invert_coord(coord);
        let source = self.dest.source.write().await;
        source.write_value(txn_id, source_coord, value).await
    }
}

#[derive(Clone)]
pub struct DenseSparse<S> {
    source: S,
    block_size: usize,
}

impl<S: SparseInstance + fmt::Debug> From<S> for DenseSparse<S> {
    fn from(source: S) -> Self {
        debug_assert!(source.shape().validate().is_ok());
        let (block_size, _) = ideal_block_size_for(source.shape());
        Self { source, block_size }
    }
}

impl<S: TensorInstance> TensorInstance for DenseSparse<S> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
        self.source.shape()
    }
}

#[async_trait]
impl<S: SparseInstance + Clone> DenseInstance for DenseSparse<S> {
    type Block = ArrayBuf<Buffer<S::DType>>;
    type DType = S::DType;

    fn block_size(&self) -> usize {
        self.block_size
    }

    async fn read_block(&self, txn_id: TxnId, block_id: u64) -> TCResult<Self::Block> {
        trace!("read block {block_id} of {self:?}");

        let block_size = self.block_size as u64;
        let offset = block_id * block_size;
        let block_axis = block_axis_for(self.shape(), self.block_size);
        let mut block_shape = block_shape_for(block_axis, self.shape(), self.block_size());

        if offset + block_size > self.size() {
            let last_block_size = (self.size() % block_size) as usize;
            block_shape[0] = last_block_size / block_shape.iter().skip(1).product::<usize>();
        }

        let range = self
            .shape()
            .iter()
            .copied()
            .enumerate()
            .map(|(x, dim)| {
                let i = if dim == 1 {
                    0
                } else {
                    let stride = self
                        .shape()
                        .iter()
                        .rev()
                        .take(self.ndim() - 1 - x)
                        .product::<u64>();

                    (offset / stride) % dim
                };

                AxisRange::At(i)
            })
            .take(block_axis)
            .chain(
                block_shape
                    .iter()
                    .copied()
                    .map(|dim| AxisRange::In(0..(dim as u64), 1)),
            )
            .collect::<Range>();

        debug_assert_eq!(range.len(), self.ndim());

        let elements = self
            .source
            .clone()
            .elements(txn_id, range.clone(), Axes::default())
            .await?;

        let values = ValueStream::new(elements, range, S::DType::zero());
        let block = values.try_collect().await?;

        ArrayBuf::new(block_shape, block).map_err(TCError::from)
    }

    async fn read_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::Block>> {
        let block_axis = block_axis_for(self.shape(), self.block_size());
        let block_shape = block_shape_for(block_axis, self.shape(), self.block_size());

        let range = Range::all(self.shape());
        let order = Axes::default();
        let elements = self.source.elements(txn_id, range.clone(), order).await?;
        let values = ValueStream::new(elements, range, S::DType::zero());
        let blocks = values
            .try_chunks(self.block_size)
            .map_err(|cause| bad_request!("dense conversion error: {}", cause))
            .map(move |result| {
                result.and_then(|block| {
                    debug_assert_eq!(
                        block.len() % block_shape.iter().skip(1).product::<usize>(),
                        0
                    );

                    let mut block_shape = block_shape.to_vec();
                    block_shape[0] = block.len() / block_shape.iter().skip(1).product::<usize>();
                    ArrayBuf::new(block_shape, block).map_err(TCError::from)
                })
            });

        Ok(Box::pin(blocks))
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> TCResult<Self::DType> {
        self.source.read_value(txn_id, coord).await
    }
}

#[async_trait]
impl<S: TensorPermitRead> TensorPermitRead for DenseSparse<S> {
    async fn read_permit(
        &self,
        txn_id: TxnId,
        range: Range,
    ) -> TCResult<SmallVec<[PermitRead<Range>; 16]>> {
        self.source.read_permit(txn_id, range).await
    }
}

impl<Txn, FE, S, T> From<DenseSparse<S>> for DenseAccess<Txn, FE, T>
where
    S: Into<SparseAccess<Txn, FE, T>>,
    T: CType,
{
    fn from(sparse: DenseSparse<S>) -> Self {
        Self::Sparse(Box::new(DenseSparse {
            source: sparse.source.into(),
            block_size: sparse.block_size,
        }))
    }
}

impl<S: fmt::Debug> fmt::Debug for DenseSparse<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "dense view of {:?}", self.source)
    }
}

#[derive(Clone)]
pub struct DenseTranspose<S> {
    source: S,
    transform: Transpose,
    block_map: ArrayBuf<StackVec<u64>>,
    block_axes: Axes,
}

impl<S: DenseInstance> DenseTranspose<S> {
    pub fn new(source: S, permutation: Option<Axes>) -> TCResult<Self> {
        let transform = Transpose::new(source.shape().clone(), permutation)?;

        let num_blocks = div_ceil(source.size(), source.block_size() as u64);
        let block_axis = block_axis_for(source.shape(), source.block_size());

        let map_shape = source
            .shape()
            .iter()
            .take(block_axis)
            .copied()
            .map(|dim| dim as usize)
            .collect();

        let valid = if num_blocks == 1 {
            assert_eq!(block_axis, 0);
            true
        } else {
            let valid_map = transform.axes()[..block_axis]
                .iter()
                .copied()
                .all(|x| x < block_axis);

            let valid_block = transform.axes()[block_axis..]
                .iter()
                .copied()
                .all(|x| x > block_axis);

            valid_map && valid_block
        };

        let (map_axes, block_axes) = if valid {
            let (map_axes, block_axes) = transform.axes().split_at(block_axis);

            let block_axes = block_axes
                .iter()
                .copied()
                .map(|x| x - block_axis)
                .collect::<Axes>();

            Ok((map_axes.to_vec(), block_axes))
        } else {
            Err(bad_request!(
                "cannot transpose axes {axes:?} of {source:?} without copying",
                axes = transform.axes()
            ))
        }?;

        let block_map = ArrayBuf::new(map_shape, (0..num_blocks).into_iter().collect())?;

        let block_map = if num_blocks > 1 {
            let block_map = block_map.transpose(Some(map_axes.to_vec()))?;
            ArrayBuf::copy(&block_map).map_err(TCError::from)
        } else {
            Ok(block_map)
        }?;

        Ok(Self {
            source,
            transform,
            block_map,
            block_axes,
        })
    }
}

impl<S: TensorInstance> TensorInstance for DenseTranspose<S> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
        self.transform.shape()
    }
}

#[async_trait]
impl<S: DenseInstance + Clone> DenseInstance for DenseTranspose<S>
where
    S::Block: NDArrayTransform,
    <S::Block as NDArrayTransform>::Transpose:
        NDArrayRead<DType = S::DType> + NDArrayTransform + Into<Array<S::DType>>,
{
    type Block = <S::Block as NDArrayTransform>::Transpose;
    type DType = S::DType;

    fn block_size(&self) -> usize {
        self.source.block_size()
    }

    async fn read_block(&self, txn_id: TxnId, block_id: u64) -> TCResult<Self::Block> {
        let source_block_id = source_block_id_for(&self.block_map, block_id)?;
        let block = self.source.read_block(txn_id, source_block_id).await?;

        block
            .transpose(Some(self.transform.axes().to_vec()))
            .map_err(TCError::from)
    }

    async fn read_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::Block>> {
        let block_axes = self.block_axes;

        let blocks = futures::stream::iter(self.block_map.into_inner())
            .map(move |block_id| {
                let source = self.source.clone();
                async move { source.read_block(txn_id, block_id).await }
            })
            .buffered(num_cpus::get())
            .map(move |result| {
                let block = result?;

                block
                    .transpose(Some(block_axes.to_vec()))
                    .map_err(TCError::from)
            });

        Ok(Box::pin(blocks))
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> TCResult<Self::DType> {
        self.shape().validate_coord(&coord)?;
        let coord = self.transform.invert_coord(coord);
        self.source.read_value(txn_id, coord).await
    }
}

#[async_trait]
impl<S: TensorPermitRead> TensorPermitRead for DenseTranspose<S> {
    async fn read_permit(
        &self,
        txn_id: TxnId,
        range: Range,
    ) -> TCResult<SmallVec<[PermitRead<Range>; 16]>> {
        self.transform.shape().validate_range(&range)?;
        let range = self.transform.invert_range(&range);
        self.source.read_permit(txn_id, range).await
    }
}

impl<Txn, FE, S, T> From<DenseTranspose<S>> for DenseAccess<Txn, FE, T>
where
    T: CType,
    S: Into<DenseAccess<Txn, FE, T>>,
{
    fn from(transpose: DenseTranspose<S>) -> Self {
        Self::Transpose(Box::new(DenseTranspose {
            source: transpose.source.into(),
            transform: transpose.transform,
            block_map: transpose.block_map,
            block_axes: transpose.block_axes,
        }))
    }
}

impl<S: fmt::Debug> fmt::Debug for DenseTranspose<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "transpose axes {:?} of {:?}",
            self.transform.axes(),
            self.source
        )
    }
}

#[derive(Clone)]
pub struct DenseUnary<S, T: CType> {
    source: S,
    block_op: fn(Array<T>) -> TCResult<Array<T>>,
    value_op: fn(T) -> T,
}

impl<S: DenseInstance> DenseUnary<S, S::DType> {
    fn new(
        source: S,
        block_op: fn(Array<S::DType>) -> TCResult<Array<S::DType>>,
        value_op: fn(S::DType) -> S::DType,
    ) -> Self {
        Self {
            source,
            block_op,
            value_op,
        }
    }

    pub fn abs(source: S) -> Self {
        Self::new(
            source,
            |block| block.abs().map(Array::from).map_err(TCError::from),
            S::DType::abs,
        )
    }

    pub fn exp(source: S) -> Self {
        Self::new(
            source,
            |block| block.exp().map(Array::from).map_err(TCError::from),
            |n| S::DType::from_float(n.to_float().exp()),
        )
    }

    pub fn ln(source: S) -> Self {
        Self::new(
            source,
            |block| block.ln().map(Array::from).map_err(TCError::from),
            |n| S::DType::from_float(n.to_float().ln()),
        )
    }

    pub fn round(source: S) -> Self {
        Self::new(
            source,
            |block| block.round().map(Array::from).map_err(TCError::from),
            S::DType::round,
        )
    }
}

impl<S: TensorInstance, T: CType> TensorInstance for DenseUnary<S, T> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn shape(&self) -> &Shape {
        self.source.shape()
    }
}

#[async_trait]
impl<S: DenseInstance> DenseInstance for DenseUnary<S, S::DType>
where
    Array<S::DType>: From<S::Block>,
{
    type Block = Array<S::DType>;
    type DType = S::DType;

    fn block_size(&self) -> usize {
        self.source.block_size()
    }

    async fn read_block(&self, txn_id: TxnId, block_id: u64) -> TCResult<Self::Block> {
        self.source
            .read_block(txn_id, block_id)
            .map_ok(Array::from)
            .map(move |result| result.and_then(|block| (self.block_op)(block)))
            .await
    }

    async fn read_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::Block>> {
        let source_blocks = self.source.read_blocks(txn_id).await?;
        let blocks = source_blocks
            .map_ok(Array::from)
            .map(move |result| result.and_then(|block| (self.block_op)(block)));

        Ok(Box::pin(blocks))
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> TCResult<Self::DType> {
        let source_value = self.source.read_value(txn_id, coord).await?;
        Ok((self.value_op)(source_value))
    }
}

#[async_trait]
impl<S: TensorPermitRead, T: CType> TensorPermitRead for DenseUnary<S, T> {
    async fn read_permit(
        &self,
        txn_id: TxnId,
        range: Range,
    ) -> TCResult<SmallVec<[PermitRead<Range>; 16]>> {
        self.source.read_permit(txn_id, range).await
    }
}

impl<Txn, FE, S, T> From<DenseUnary<S, T>> for DenseAccess<Txn, FE, T>
where
    S: Into<DenseAccess<Txn, FE, T>>,
    T: CType,
{
    fn from(unary: DenseUnary<S, T>) -> Self {
        Self::Unary(Box::new(DenseUnary {
            source: unary.source.into(),
            block_op: unary.block_op,
            value_op: unary.value_op,
        }))
    }
}

impl<S: fmt::Debug, T: CType + DType> fmt::Debug for DenseUnary<S, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "unary transform of {:?}", self.source)
    }
}

pub struct DenseUnaryCast<Txn, FE, T: CType> {
    source: DenseAccessCast<Txn, FE>,
    block_op: fn(Block) -> TCResult<Array<T>>,
    value_op: fn(Number) -> T,
}

impl<Txn, FE, T: CType> Clone for DenseUnaryCast<Txn, FE, T> {
    fn clone(&self) -> Self {
        Self {
            source: self.source.clone(),
            block_op: self.block_op,
            value_op: self.value_op,
        }
    }
}

impl<Txn, FE, T: CType> DenseUnaryCast<Txn, FE, T> {
    pub fn new<S>(
        source: S,
        block_op: fn(Block) -> TCResult<Array<T>>,
        value_op: fn(Number) -> T,
    ) -> Self
    where
        S: Into<DenseAccessCast<Txn, FE>>,
    {
        Self {
            source: source.into(),
            block_op,
            value_op,
        }
    }
}

macro_rules! block_f32_cast {
    ($this:ident, $var:ident, $call:expr) => {
        match $this {
            Block::F32($var) => $call,
            Block::I16($var) => $call,
            Block::I32($var) => $call,
            Block::U8($var) => $call,
            Block::U16($var) => $call,
            Block::U32($var) => $call,
            block => unreachable!("32-bit float op on {:?}", block),
        }
    };
}

impl<Txn, FE> DenseUnaryCast<Txn, FE, f32> {
    pub fn asin_f32<S: Into<DenseAccessCast<Txn, FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            block_op: |block| {
                block_f32_cast!(
                    block,
                    array,
                    array.asin().map(Array::from).map_err(TCError::from)
                )
            },
            value_op: |n| n.asin().cast_into(),
        }
    }

    pub fn sin_f32<S: Into<DenseAccessCast<Txn, FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            block_op: |block| {
                block_f32_cast!(
                    block,
                    array,
                    array.sin().map(Array::from).map_err(TCError::from)
                )
            },
            value_op: |n| n.sin().cast_into(),
        }
    }

    pub fn sinh_f32<S: Into<DenseAccessCast<Txn, FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            block_op: |block| {
                block_f32_cast!(
                    block,
                    array,
                    array.sinh().map(Array::from).map_err(TCError::from)
                )
            },
            value_op: |n| n.sinh().cast_into(),
        }
    }

    pub fn acos_f32<S: Into<DenseAccessCast<Txn, FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            block_op: |block| {
                block_f32_cast!(
                    block,
                    array,
                    array.acos().map(Array::from).map_err(TCError::from)
                )
            },
            value_op: |n| n.acos().cast_into(),
        }
    }

    pub fn cos_f32<S: Into<DenseAccessCast<Txn, FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            block_op: |block| {
                block_f32_cast!(
                    block,
                    array,
                    array.cos().map(Array::from).map_err(TCError::from)
                )
            },
            value_op: |n| n.cos().cast_into(),
        }
    }

    pub fn cosh_f32<S: Into<DenseAccessCast<Txn, FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            block_op: |block| {
                block_f32_cast!(
                    block,
                    array,
                    array.cosh().map(Array::from).map_err(TCError::from)
                )
            },
            value_op: |n| n.cosh().cast_into(),
        }
    }

    pub fn atan_f32<S: Into<DenseAccessCast<Txn, FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            block_op: |block| {
                block_f32_cast!(
                    block,
                    array,
                    array.atan().map(Array::from).map_err(TCError::from)
                )
            },
            value_op: |n| n.atan().cast_into(),
        }
    }

    pub fn tan_f32<S: Into<DenseAccessCast<Txn, FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            block_op: |block| {
                block_f32_cast!(
                    block,
                    array,
                    array.tan().map(Array::from).map_err(TCError::from)
                )
            },
            value_op: |n| n.tan().cast_into(),
        }
    }

    pub fn tanh_f32<S: Into<DenseAccessCast<Txn, FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            block_op: |block| {
                block_f32_cast!(
                    block,
                    array,
                    array.tanh().map(Array::from).map_err(TCError::from)
                )
            },
            value_op: |n| n.tanh().cast_into(),
        }
    }
}

macro_rules! block_f64_cast {
    ($this:ident, $var:ident, $call:expr) => {
        match $this {
            Block::F64($var) => $call,
            Block::I64($var) => $call,
            Block::U64($var) => $call,
            block => unreachable!("64-bit float op on {:?}", block),
        }
    };
}

impl<Txn, FE> DenseUnaryCast<Txn, FE, f64> {
    pub fn asin_f64<S: Into<DenseAccessCast<Txn, FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            block_op: |block| {
                block_f64_cast!(
                    block,
                    array,
                    array.asin().map(Array::from).map_err(TCError::from)
                )
            },
            value_op: |n| n.asin().cast_into(),
        }
    }

    pub fn sin_f64<S: Into<DenseAccessCast<Txn, FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            block_op: |block| {
                block_f64_cast!(
                    block,
                    array,
                    array.sin().map(Array::from).map_err(TCError::from)
                )
            },
            value_op: |n| n.sin().cast_into(),
        }
    }

    pub fn sinh_f64<S: Into<DenseAccessCast<Txn, FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            block_op: |block| {
                block_f64_cast!(
                    block,
                    array,
                    array.sinh().map(Array::from).map_err(TCError::from)
                )
            },
            value_op: |n| n.sinh().cast_into(),
        }
    }

    pub fn acos_f64<S: Into<DenseAccessCast<Txn, FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            block_op: |block| {
                block_f64_cast!(
                    block,
                    array,
                    array.acos().map(Array::from).map_err(TCError::from)
                )
            },
            value_op: |n| n.acos().cast_into(),
        }
    }

    pub fn cos_f64<S: Into<DenseAccessCast<Txn, FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            block_op: |block| {
                block_f64_cast!(
                    block,
                    array,
                    array.cos().map(Array::from).map_err(TCError::from)
                )
            },
            value_op: |n| n.cos().cast_into(),
        }
    }

    pub fn cosh_f64<S: Into<DenseAccessCast<Txn, FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            block_op: |block| {
                block_f64_cast!(
                    block,
                    array,
                    array.cosh().map(Array::from).map_err(TCError::from)
                )
            },
            value_op: |n| n.cosh().cast_into(),
        }
    }

    pub fn atan_f64<S: Into<DenseAccessCast<Txn, FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            block_op: |block| {
                block_f64_cast!(
                    block,
                    array,
                    array.atan().map(Array::from).map_err(TCError::from)
                )
            },
            value_op: |n| n.atan().cast_into(),
        }
    }

    pub fn tan_f64<S: Into<DenseAccessCast<Txn, FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            block_op: |block| {
                block_f64_cast!(
                    block,
                    array,
                    array.tan().map(Array::from).map_err(TCError::from)
                )
            },
            value_op: |n| n.tan().cast_into(),
        }
    }

    pub fn tanh_f64<S: Into<DenseAccessCast<Txn, FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            block_op: |block| {
                block_f64_cast!(
                    block,
                    array,
                    array.tanh().map(Array::from).map_err(TCError::from)
                )
            },
            value_op: |n| n.tanh().cast_into(),
        }
    }
}

impl<Txn, FE> DenseUnaryCast<Txn, FE, u8> {
    pub fn not<S: Into<DenseAccessCast<Txn, FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            block_op: Block::not,
            value_op: |n| if bool::cast_from(n.not()) { 1 } else { 0 },
        }
    }
}

impl<Txn, FE, T> TensorInstance for DenseUnaryCast<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CType + DType,
{
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        let source = &self.source;
        cast_dispatch!(source, this, this.shape())
    }
}

#[async_trait]
impl<Txn, FE, T> DenseInstance for DenseUnaryCast<Txn, FE, T>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<Node>,
    T: CType + DType + fmt::Debug,
    Number: From<T> + CastInto<T>,
{
    type Block = Array<T>;
    type DType = T;

    fn block_size(&self) -> usize {
        let source = &self.source;
        cast_dispatch!(source, this, this.block_size())
    }

    async fn read_block(&self, txn_id: TxnId, block_id: u64) -> TCResult<Self::Block> {
        self.source
            .read_block(txn_id, block_id)
            .map(|result| result.and_then(|block| (self.block_op)(block)))
            .await
    }

    async fn read_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::Block>> {
        let source_blocks = self.source.read_blocks(txn_id).await?;
        let blocks =
            source_blocks.map(move |result| result.and_then(|block| (self.block_op)(block)));

        Ok(Box::pin(blocks))
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> TCResult<Self::DType> {
        let source_value = self.source.read_value(txn_id, coord).await?;
        Ok((self.value_op)(source_value))
    }
}

#[async_trait]
impl<Txn, FE, T> TensorPermitRead for DenseUnaryCast<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CType,
{
    async fn read_permit(
        &self,
        txn_id: TxnId,
        range: Range,
    ) -> TCResult<SmallVec<[PermitRead<Range>; 16]>> {
        let source = &self.source;
        cast_dispatch!(source, this, this.read_permit(txn_id, range).await)
    }
}

impl<Txn, FE, T: CType> From<DenseUnaryCast<Txn, FE, T>> for DenseAccess<Txn, FE, T> {
    fn from(unary: DenseUnaryCast<Txn, FE, T>) -> Self {
        Self::UnaryCast(Box::new(unary))
    }
}

impl<Txn, FE, T> fmt::Debug for DenseUnaryCast<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CType + DType,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "unary transform/cast of {:?}", self.source)
    }
}

pub struct DenseVersion<FE, T> {
    file: DenseFile<FE, T>,
    semaphore: Semaphore,
}

impl<FE, T> Clone for DenseVersion<FE, T> {
    fn clone(&self) -> Self {
        Self {
            file: self.file.clone(),
            semaphore: self.semaphore.clone(),
        }
    }
}
impl<FE, T> DenseVersion<FE, T> {
    pub fn new(file: DenseFile<FE, T>, semaphore: Semaphore) -> Self {
        Self { file, semaphore }
    }
}

impl<FE, T> DenseVersion<FE, T>
where
    FE: DenseCacheFile + AsType<Buffer<T>> + for<'en> fs::FileSave<'en>,
    T: CType + DType,
    Buffer<T>: de::FromStream<Context = ()>,
{
    pub fn commit(&self, txn_id: &TxnId) {
        self.semaphore.finalize(txn_id, true)
    }

    pub async fn finalize(&self, txn_id: &TxnId) {
        self.file.commit().await;
        self.semaphore.finalize(txn_id, true)
    }

    pub fn rollback(&self, txn_id: &TxnId) {
        self.semaphore.finalize(txn_id, false)
    }
}

impl<FE, T> TensorInstance for DenseVersion<FE, T>
where
    FE: ThreadSafe,
    T: DType + ThreadSafe,
{
    fn dtype(&self) -> NumberType {
        self.file.dtype()
    }

    fn shape(&self) -> &Shape {
        self.file.shape()
    }
}

#[async_trait]
impl<FE, T> DenseInstance for DenseVersion<FE, T>
where
    FE: AsType<Buffer<T>> + ThreadSafe,
    T: CType + DType + 'static,
    Buffer<T>: de::FromStream<Context = ()>,
{
    type Block = <DenseFile<FE, T> as DenseInstance>::Block;
    type DType = <DenseFile<FE, T> as DenseInstance>::DType;

    fn block_size(&self) -> usize {
        self.file.block_size()
    }

    async fn read_block(&self, txn_id: TxnId, block_id: u64) -> TCResult<Self::Block> {
        self.file.read_block(txn_id, block_id).await
    }

    async fn read_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::Block>> {
        self.file.read_blocks(txn_id).await
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> TCResult<Self::DType> {
        self.file.read_value(txn_id, coord).await
    }
}

#[async_trait]
impl<FE, T> TensorPermitRead for DenseVersion<FE, T>
where
    FE: Send + Sync,
    T: CType + DType,
{
    async fn read_permit(
        &self,
        txn_id: TxnId,
        range: Range,
    ) -> TCResult<SmallVec<[PermitRead<Range>; 16]>> {
        self.semaphore
            .read(txn_id, range)
            .map_ok(|permit| smallvec![permit])
            .map_err(TCError::from)
            .await
    }
}

#[async_trait]
impl<FE, T> TensorPermitWrite for DenseVersion<FE, T>
where
    FE: Send + Sync,
    T: CType + DType,
{
    async fn write_permit(&self, txn_id: TxnId, range: Range) -> TCResult<PermitWrite<Range>> {
        self.semaphore
            .write(txn_id, range)
            .map_err(TCError::from)
            .await
    }
}

#[async_trait]
impl<'a, FE, T> DenseWriteLock<'a> for DenseVersion<FE, T>
where
    FE: AsType<Buffer<T>> + ThreadSafe,
    T: CType + DType,
    Buffer<T>: de::FromStream<Context = ()>,
{
    type WriteGuard = <DenseFile<FE, T> as DenseWriteLock<'a>>::WriteGuard;

    async fn write(&'a self) -> Self::WriteGuard {
        self.file.write().await
    }
}

impl<Txn, FE, T: CType> From<DenseVersion<FE, T>> for DenseAccess<Txn, FE, T> {
    fn from(version: DenseVersion<FE, T>) -> Self {
        Self::Version(version)
    }
}

impl<FE, T> fmt::Debug for DenseVersion<FE, T>
where
    FE: ThreadSafe,
    T: CType + DType,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "transactional version of {:?}", self.file)
    }
}

#[inline]
fn source_block_id_for(block_map: &ArrayBuf<StackVec<u64>>, block_id: u64) -> TCResult<u64> {
    block_map
        .as_slice()
        .get(block_id as usize)
        .copied()
        .ok_or_else(|| bad_request!("block id {} is out of range", block_id))
}
