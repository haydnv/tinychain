use std::cmp::Ordering;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use std::{fmt, io};

use async_trait::async_trait;
use collate::Collate;
use destream::de;
use freqfs::*;
use futures::future::{Future, FutureExt, TryFutureExt};
use futures::stream::{self, Stream, StreamExt, TryStreamExt};
use futures::try_join;
use ha_ndarray::*;
use safecast::{AsType, CastFrom, CastInto};

use tc_error::*;
use tc_transact::lock::{PermitRead, PermitWrite};
use tc_transact::{Transaction, TxnId};
use tc_value::{
    DType, Number, NumberClass, NumberCollator, NumberInstance, NumberType, Trigonometry,
};
use tcgeneric::ThreadSafe;

use super::base::DenseBase;
use super::{
    div_ceil, ideal_block_size_for, DenseInstance, DenseWrite, DenseWriteGuard, DenseWriteLock,
};

use crate::tensor::block::Block;
use crate::tensor::sparse::{Node, SparseAccess, SparseInstance};
use crate::tensor::transform::{Broadcast, Expand, Reduce, Reshape, Slice, Transpose};
use crate::tensor::{
    coord_of, offset_of, Axes, AxisRange, Coord, Range, Semaphore, Shape, TensorInstance,
    TensorPermitRead, TensorPermitWrite,
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
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        cast_dispatch!(self, this, this.read_permit(txn_id, range).await)
    }
}

impl<Txn, FE> fmt::Debug for DenseAccessCast<Txn, FE> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        cast_dispatch!(self, this, this.fmt(f))
    }
}

pub enum DenseAccess<Txn, FE, T: CDatatype> {
    Base(DenseBase<Txn, FE, T>),
    File(DenseFile<FE, T>),
    Broadcast(Box<DenseBroadcast<Self>>),
    Combine(Box<DenseCombine<Self, Self, T>>),
    CombineConst(Box<DenseCombineConst<Self, T>>),
    Compare(Box<DenseCompare<Txn, FE, T>>),
    CompareConst(Box<DenseCompareConst<Txn, FE, T>>),
    Const(Box<DenseConst<Self, T>>),
    Cow(Box<DenseCow<FE, Self>>),
    Diagonal(Box<DenseDiagonal<Self>>),
    Expand(Box<DenseExpand<Self>>),
    Reduce(Box<DenseReduce<Self, T>>),
    Reshape(Box<DenseReshape<Self>>),
    Slice(Box<DenseSlice<Self>>),
    Sparse(DenseSparse<SparseAccess<Txn, FE, T>>),
    Transpose(Box<DenseTranspose<Self>>),
    Unary(Box<DenseUnary<Self, T>>),
    UnaryCast(Box<DenseUnaryCast<Txn, FE, T>>),
    Version(DenseVersion<FE, T>),
}

impl<Txn, FE, T: CDatatype> Clone for DenseAccess<Txn, FE, T> {
    fn clone(&self) -> Self {
        match self {
            Self::Base(base) => Self::Base(base.clone()),
            Self::File(file) => Self::File(file.clone()),
            Self::Broadcast(broadcast) => Self::Broadcast(broadcast.clone()),
            Self::Combine(combine) => Self::Combine(combine.clone()),
            Self::CombineConst(combine) => Self::CombineConst(combine.clone()),
            Self::Compare(compare) => Self::Compare(compare.clone()),
            Self::CompareConst(compare) => Self::CompareConst(compare.clone()),
            Self::Const(combine) => Self::Const(combine.clone()),
            Self::Cow(cow) => Self::Cow(cow.clone()),
            Self::Diagonal(diag) => Self::Diagonal(diag.clone()),
            Self::Expand(expand) => Self::Expand(expand.clone()),
            Self::Reduce(reduce) => Self::Reduce(reduce.clone()),
            Self::Reshape(reshape) => Self::Reshape(reshape.clone()),
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
            DenseAccess::Const($var) => $call,
            DenseAccess::Cow($var) => $call,
            DenseAccess::Diagonal($var) => $call,
            DenseAccess::Expand($var) => $call,
            DenseAccess::Reduce($var) => $call,
            DenseAccess::Reshape($var) => $call,
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
    T: CDatatype + DType,
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
    T: CDatatype + DType + fmt::Debug,
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
            Self::Const(combine) => combine.read_blocks(txn_id).await,
            Self::Cow(cow) => cow.read_blocks(txn_id).await,
            Self::Diagonal(diag) => Ok(Box::pin(
                diag.read_blocks(txn_id).await?.map_ok(Array::from),
            )),
            Self::Expand(expand) => Ok(Box::pin(
                expand.read_blocks(txn_id).await?.map_ok(Array::from),
            )),
            Self::Reduce(reduce) => reduce.read_blocks(txn_id).await,
            Self::Reshape(reshape) => Ok(Box::pin(
                reshape.read_blocks(txn_id).await?.map_ok(Array::from),
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
    T: CDatatype + DType,
{
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        match self {
            Self::Base(base) => base.read_permit(txn_id, range).await,
            Self::Broadcast(broadcast) => broadcast.read_permit(txn_id, range).await,
            Self::Combine(combine) => combine.read_permit(txn_id, range).await,
            Self::CombineConst(combine) => combine.read_permit(txn_id, range).await,
            Self::Compare(compare) => compare.read_permit(txn_id, range).await,
            Self::CompareConst(compare) => compare.read_permit(txn_id, range).await,
            Self::Const(combine) => combine.read_permit(txn_id, range).await,
            Self::Cow(cow) => cow.read_permit(txn_id, range).await,
            Self::Diagonal(diag) => diag.read_permit(txn_id, range).await,
            Self::Expand(expand) => expand.read_permit(txn_id, range).await,
            Self::Reshape(reshape) => reshape.read_permit(txn_id, range).await,
            Self::Slice(slice) => slice.read_permit(txn_id, range).await,
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
    Txn: Send + Sync,
    FE: Send + Sync,
    T: CDatatype + DType,
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

impl<Txn, FE, T: CDatatype + DType> fmt::Debug for DenseAccess<Txn, FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        access_dispatch!(self, this, this.fmt(f))
    }
}

pub struct DenseFile<FE, T> {
    dir: DirLock<FE>,
    block_map: ArrayBase<Vec<u64>>,
    block_size: usize,
    shape: Shape,
    dtype: PhantomData<T>,
}

impl<FE, T> Clone for DenseFile<FE, T> {
    fn clone(&self) -> Self {
        Self {
            dir: self.dir.clone(),
            block_map: self.block_map.clone(),
            block_size: self.block_size,
            shape: self.shape.clone(),
            dtype: PhantomData,
        }
    }
}

impl<FE, T> DenseFile<FE, T>
where
    FE: DenseCacheFile + AsType<Buffer<T>>,
    T: CDatatype + DType + NumberInstance,
    Buffer<T>: de::FromStream<Context = ()>,
{
    pub async fn load(dir: DirLock<FE>, shape: Shape) -> TCResult<Self> {
        let contents = dir.read().await;
        let num_blocks = contents.len();

        if num_blocks == 0 {
            return Err(bad_request!(
                "cannot load a dense tensor from an empty directory"
            ));
        }

        let mut size = 0u64;

        let block_size = {
            let block = contents
                .get_file(&0)
                .ok_or_else(|| TCError::not_found("block 0"))?;

            let block: FileReadGuard<Buffer<T>> = block.read().await?;
            size += block.len() as u64;
            block.len()
        };

        let block_axis = block_axis_for(&shape, block_size);
        let block_shape = block_shape_for(block_axis, &shape, block_size);

        for block_id in 1..(num_blocks - 1) {
            let block = contents
                .get_file(&block_id)
                .ok_or_else(|| TCError::not_found(format!("block {block_id}")))?;

            let block: FileReadGuard<Buffer<T>> = block.read().await?;
            if block.len() == block_size {
                size += block.len() as u64;
            } else {
                return Err(bad_request!(
                    "block {} has incorrect size {} (expected {})",
                    block_id,
                    block.len(),
                    block_size
                ));
            }
        }

        {
            let block_id = num_blocks - 1;
            let block = contents
                .get_file(&block_id)
                .ok_or_else(|| bad_request!("block {block_id}"))?;

            let block: FileReadGuard<Buffer<T>> = block.read().await?;
            size += block.len() as u64;
        }

        std::mem::drop(contents);

        if size != shape.size() {
            return Err(bad_request!(
                "tensor blocks have incorrect total length {} (expected {} for shape {:?})",
                size,
                shape.size(),
                shape
            ));
        }

        let mut block_map_shape = BlockShape::with_capacity(block_axis + 1);
        block_map_shape.extend(
            shape
                .iter()
                .take(block_axis)
                .copied()
                .map(|dim| dim as usize),
        );
        block_map_shape.push(shape[block_axis] as usize / block_shape[0]);

        let block_map = ArrayBase::<Vec<_>>::new(
            block_map_shape,
            (0..num_blocks as u64).into_iter().collect(),
        )?;

        Ok(Self {
            dir,
            block_map,
            block_size,
            shape,
            dtype: PhantomData,
        })
    }

    pub async fn constant(dir: DirLock<FE>, shape: Shape, value: T) -> TCResult<Self> {
        shape.validate()?;

        let size = shape.iter().product::<u64>();

        let (block_size, num_blocks) = ideal_block_size_for(&shape);

        debug_assert!(block_size > 0);

        {
            let dtype_size = T::dtype().size();

            let mut dir = dir.write().await;

            for block_id in 0..num_blocks {
                dir.create_file::<Buffer<T>>(
                    block_id.to_string(),
                    vec![value; block_size].into(),
                    block_size * dtype_size,
                )?;
            }

            let last_block_id = num_blocks - 1;
            if size % block_size as u64 == 0 {
                dir.create_file::<Buffer<T>>(
                    last_block_id.to_string(),
                    vec![value; block_size].into(),
                    block_size * dtype_size,
                )
            } else {
                dir.create_file::<Buffer<T>>(
                    last_block_id.to_string(),
                    vec![value; block_size].into(),
                    block_size * dtype_size,
                )
            }?;
        };

        let block_axis = block_axis_for(&shape, block_size);
        let map_shape = shape
            .iter()
            .take(block_axis)
            .copied()
            .map(|dim| dim as usize)
            .collect();

        let block_map = (0u64..num_blocks as u64).into_iter().collect();
        let block_map = ArrayBase::<Vec<_>>::new(map_shape, block_map)?;

        Ok(Self {
            dir,
            block_map,
            block_size,
            shape,
            dtype: PhantomData,
        })
    }
}

impl<FE, T> TensorInstance for DenseFile<FE, T>
where
    FE: ThreadSafe,
    T: DType + ThreadSafe,
{
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        &self.shape
    }
}

#[async_trait]
impl<FE, T> DenseInstance for DenseFile<FE, T>
where
    FE: FileLoad + AsType<Buffer<T>>,
    T: CDatatype + DType + 'static,
    Buffer<T>: de::FromStream<Context = ()>,
{
    type Block = ArrayBase<FileReadGuardOwned<FE, Buffer<T>>>;
    type DType = T;

    fn block_size(&self) -> usize {
        self.block_size
    }

    async fn read_block(&self, _txn_id: TxnId, block_id: u64) -> TCResult<Self::Block> {
        let dir = self.dir.read().await;
        let file = dir.get_file(&block_id).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("dense tensor block {}", block_id),
            )
        })?;

        let buffer = file.read_owned().await?;
        let block_axis = block_axis_for(self.shape(), self.block_size);
        let block_shape = block_shape_for(block_axis, &self.shape, buffer.len());
        ArrayBase::<FileReadGuardOwned<FE, Buffer<T>>>::new(block_shape, buffer)
            .map_err(TCError::from)
    }

    async fn read_blocks(self, _txn_id: TxnId) -> TCResult<BlockStream<Self::Block>> {
        let shape = self.shape;
        let block_axis = block_axis_for(&shape, self.block_size);
        let dir = self.dir.into_read().await;

        let blocks = stream::iter(self.block_map.into_inner())
            .map(move |block_id| {
                dir.get_file(&block_id).cloned().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::NotFound,
                        format!("dense tensor block {}", block_id),
                    )
                    .into()
                })
            })
            .map_ok(|block| block.into_read())
            .try_buffered(num_cpus::get())
            .map(move |result| {
                let buffer = result?;
                let block_shape = block_shape_for(block_axis, &shape, buffer.len());
                ArrayBase::<FileReadGuardOwned<FE, Buffer<T>>>::new(block_shape, buffer)
                    .map_err(TCError::from)
            });

        Ok(Box::pin(blocks))
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> TCResult<Self::DType> {
        self.shape().validate_coord(&coord)?;

        let offset = offset_of(coord, self.shape());
        let block_id = offset / self.block_size() as u64;
        let block_offset = (offset % self.block_size() as u64) as usize;

        let block = self.read_block(txn_id, block_id).await?;
        let context = ha_ndarray::Context::default()?;
        let queue = ha_ndarray::Queue::new(context, self.block_size())?;
        let buffer = block.read(&queue)?;
        Ok(buffer.to_slice()?.as_ref()[block_offset])
    }
}

#[async_trait]
impl<'a, FE, T> DenseWrite for DenseFile<FE, T>
where
    FE: FileLoad + AsType<Buffer<T>>,
    T: CDatatype + DType + 'static,
    Buffer<T>: de::FromStream<Context = ()>,
{
    type BlockWrite = ArrayBase<FileWriteGuardOwned<FE, Buffer<T>>>;

    async fn write_block(&self, _txn_id: TxnId, block_id: u64) -> TCResult<Self::BlockWrite> {
        let dir = self.dir.read().await;
        let file = dir.get_file(&block_id).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("dense tensor block {}", block_id),
            )
        })?;

        let buffer = file.write_owned().await?;
        let block_axis = block_axis_for(self.shape(), self.block_size);
        let block_shape = block_shape_for(block_axis, &self.shape, buffer.len());
        ArrayBase::<FileWriteGuardOwned<FE, Buffer<T>>>::new(block_shape, buffer)
            .map_err(TCError::from)
    }

    async fn write_blocks(self, _txn_id: TxnId) -> TCResult<BlockStream<Self::BlockWrite>> {
        let shape = self.shape;
        let block_axis = block_axis_for(&shape, self.block_size);
        let dir = self.dir.into_read().await;

        let blocks = stream::iter(self.block_map.into_inner())
            .map(move |block_id| {
                dir.get_file(&block_id).cloned().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::NotFound,
                        format!("dense tensor block {}", block_id),
                    )
                    .into()
                })
            })
            .map_ok(|block| block.into_write())
            .try_buffered(num_cpus::get())
            .map(move |result| {
                let buffer = result?;
                let block_shape = block_shape_for(block_axis, &shape, buffer.len());
                ArrayBase::<FileWriteGuardOwned<FE, Buffer<T>>>::new(block_shape, buffer)
                    .map_err(TCError::from)
            });

        Ok(Box::pin(blocks))
    }
}

#[async_trait]
impl<'a, FE, T> DenseWriteLock<'a> for DenseFile<FE, T>
where
    FE: FileLoad + AsType<Buffer<T>>,
    T: CDatatype + DType,
    Buffer<T>: de::FromStream<Context = ()>,
{
    type WriteGuard = DenseFileWriteGuard<'a, FE>;

    async fn write(&'a self) -> Self::WriteGuard {
        let dir = self.dir.read().await;

        DenseFileWriteGuard {
            dir: Arc::new(dir),
            block_size: self.block_size,
            shape: &self.shape,
        }
    }
}

impl<Txn, FE, T: CDatatype> From<DenseFile<FE, T>> for DenseAccess<Txn, FE, T> {
    fn from(file: DenseFile<FE, T>) -> Self {
        Self::File(file)
    }
}

impl<FE, T> fmt::Debug for DenseFile<FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "dense tensor with shape {:?}", self.shape)
    }
}

pub struct DenseFileWriteGuard<'a, FE> {
    dir: Arc<DirReadGuard<'a, FE>>,
    block_size: usize,
    shape: &'a Shape,
}

impl<'a, FE> DenseFileWriteGuard<'a, FE> {
    pub async fn merge<T>(&self, other: DirLock<FE>) -> TCResult<()>
    where
        FE: FileLoad + AsType<Buffer<T>>,
        T: CDatatype + DType + 'static,
        Buffer<T>: de::FromStream<Context = ()>,
    {
        let num_blocks = div_ceil(self.shape.size(), self.block_size as u64);
        stream::iter(0..num_blocks)
            .map(move |block_id| {
                let that = other.clone();

                async move {
                    let that = that.read().await;
                    if that.contains(&block_id) {
                        let mut this = self.dir.write_file(&block_id).await?;
                        let that = that.read_file(&block_id).await?;
                        this.write(&*that).map_err(TCError::from)
                    } else {
                        Ok(())
                    }
                }
            })
            .buffer_unordered(num_cpus::get())
            .try_fold((), |(), _| futures::future::ready(Ok(())))
            .await
    }
}

#[async_trait]
impl<'a, FE, T> DenseWriteGuard<T> for DenseFileWriteGuard<'a, FE>
where
    FE: FileLoad + AsType<Buffer<T>>,
    T: CDatatype + DType + 'static,
    Buffer<T>: de::FromStream<Context = ()>,
{
    async fn overwrite<O: DenseInstance<DType = T>>(
        &self,
        txn_id: TxnId,
        other: O,
    ) -> TCResult<()> {
        let block_axis = block_axis_for(&self.shape, self.block_size);
        let block_shape = block_shape_for(block_axis, &self.shape, self.block_size);

        let context = ha_ndarray::Context::default()?;
        let queue = ha_ndarray::Queue::new(context, block_shape.iter().product())?;

        let blocks = other.read_blocks(txn_id).await?;
        let blocks = BlockResize::new(blocks, block_shape)?;

        blocks
            .enumerate()
            .map(|(block_id, result)| {
                let dir = self.dir.clone();
                let queue = queue.clone();

                async move {
                    let data = result?;
                    let data = data.read(&queue)?;
                    let mut block = dir.write_file(&block_id).await?;
                    debug_assert_eq!(block.len(), data.len());
                    block.write(data)?;
                    Result::<(), TCError>::Ok(())
                }
            })
            .buffered(num_cpus::get())
            .try_fold((), |(), ()| futures::future::ready(Ok(())))
            .await
    }

    async fn overwrite_value(&self, txn_id: TxnId, value: T) -> TCResult<()> {
        let num_blocks = div_ceil(self.shape.size(), self.block_size as u64);

        stream::iter(0..num_blocks)
            .map(|block_id| async move {
                let mut block = self.dir.write_file(&block_id).await?;
                block.write_value(value).map_err(TCError::from)
            })
            .buffered(num_cpus::get())
            .try_fold((), |(), _| futures::future::ready(Ok(())))
            .await
    }

    async fn write_value(&self, txn_id: TxnId, coord: Coord, value: T) -> TCResult<()> {
        self.shape.validate_coord(&coord)?;

        let offset = offset_of(coord, &self.shape);
        let block_id = offset / self.block_size as u64;

        let mut block = self.dir.write_file(&block_id).await?;
        block.write_value_at((offset % self.block_size as u64) as usize, value)?;

        Ok(())
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

    pub fn commit(&self, txn_id: &TxnId) {
        self.semaphore.finalize(txn_id, false)
    }

    pub fn rollback(&self, txn_id: &TxnId) {
        self.semaphore.finalize(txn_id, false)
    }

    pub fn finalize(&self, txn_id: &TxnId) {
        self.semaphore.finalize(txn_id, true)
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
    FE: FileLoad + AsType<Buffer<T>>,
    T: CDatatype + DType + 'static,
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
    T: CDatatype + DType,
{
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        self.semaphore
            .read(txn_id, range)
            .map_ok(|permit| vec![permit])
            .map_err(TCError::from)
            .await
    }
}

#[async_trait]
impl<FE, T> TensorPermitWrite for DenseVersion<FE, T>
where
    FE: Send + Sync,
    T: CDatatype + DType,
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
    FE: FileLoad + AsType<Buffer<T>>,
    T: CDatatype + DType,
    Buffer<T>: de::FromStream<Context = ()>,
{
    type WriteGuard = <DenseFile<FE, T> as DenseWriteLock<'a>>::WriteGuard;

    async fn write(&'a self) -> Self::WriteGuard {
        self.file.write().await
    }
}

impl<Txn, FE, T: CDatatype> From<DenseVersion<FE, T>> for DenseAccess<Txn, FE, T> {
    fn from(version: DenseVersion<FE, T>) -> Self {
        Self::Version(version)
    }
}

impl<FE, T> fmt::Debug for DenseVersion<FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "transactional version of {:?}", self.file)
    }
}

#[derive(Clone)]
pub struct DenseBroadcast<S> {
    source: S,
    transform: Broadcast,
    block_map: ArrayBase<Vec<u64>>,
    block_size: usize,
}

impl<S: DenseInstance> DenseBroadcast<S> {
    pub fn new(source: S, shape: Shape) -> TCResult<Self> {
        let transform = Broadcast::new(source.shape().clone(), shape)?;

        let num_blocks = div_ceil(source.size(), source.block_size() as u64);
        let block_axis = block_axis_for(source.shape(), source.block_size());
        let source_block_shape = block_shape_for(block_axis, source.shape(), source.block_size());

        let mut block_shape = BlockShape::with_capacity(source_block_shape.len());
        block_shape.push(source_block_shape[0]);
        block_shape.extend(
            transform
                .shape()
                .iter()
                .rev()
                .take(source_block_shape.len() - 1)
                .rev()
                .copied()
                .map(|dim| dim as usize),
        );

        let block_size = block_shape.iter().product();

        let mut block_map_shape = BlockShape::with_capacity(source.ndim());
        block_map_shape.extend(
            transform
                .shape()
                .iter()
                .take(block_axis)
                .copied()
                .map(|dim| dim as usize),
        );
        block_map_shape.push(transform.shape()[block_axis] as usize / source_block_shape[0]);

        let block_map =
            ArrayBase::<Vec<_>>::new(block_map_shape, (0..num_blocks).into_iter().collect())?;

        Ok(Self {
            source,
            transform,
            block_map,
            block_size,
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
        let source_block = self.source.read_block(txn_id, source_block_id).await?;
        source_block.broadcast(block_shape).map_err(TCError::from)
    }

    async fn read_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::Block>> {
        let block_axis = block_axis_for(self.shape(), self.block_size);
        let block_shape = block_shape_for(block_axis, self.shape(), self.block_size);

        let blocks = stream::iter(self.block_map.into_inner())
            .map(move |block_id| {
                let source = self.source.clone();
                async move { source.read_block(txn_id, block_id).await }
            })
            .buffered(num_cpus::get())
            .map(move |result| {
                let source_block = result?;
                source_block
                    .broadcast(block_shape.to_vec())
                    .map_err(TCError::from)
            });

        Ok(Box::pin(blocks))
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> TCResult<Self::DType> {
        self.shape().validate_coord(&coord)?;
        let coord = self.transform.invert_coord(&coord);
        self.source.read_value(txn_id, coord).await
    }
}

#[async_trait]
impl<S: TensorPermitRead> TensorPermitRead for DenseBroadcast<S> {
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        self.transform.shape().validate_range(&range)?;
        let range = self.transform.invert_range(range);
        self.source.read_permit(txn_id, range).await
    }
}

impl<Txn, FE, T, S> From<DenseBroadcast<S>> for DenseAccess<Txn, FE, T>
where
    T: CDatatype,
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

            let context = ha_ndarray::Context::default()?;
            let queue = ha_ndarray::Queue::new(context, block.size())?;
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
        let dir = self.dir.read().await;

        if let Some(block) = dir.get_file(&block_id) {
            let buffer: Buffer<S::DType> = block
                .read_owned::<Buffer<S::DType>>()
                .map_ok(|block| block.clone().into())
                .map_err(TCError::from)
                .await?;

            let block_axis = block_axis_for(self.shape(), self.block_size());
            let block_data_size = S::DType::dtype().size() * buffer.len();
            let block_shape = block_shape_for(block_axis, self.shape(), block_data_size);
            let block = ArrayBase::<Buffer<S::DType>>::new(block_shape, buffer)?;

            Ok(block.into())
        } else {
            self.source
                .read_block(txn_id, block_id)
                .map_ok(Array::from)
                .await
        }
    }

    async fn read_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::Block>> {
        let num_blocks = div_ceil(self.size(), self.block_size() as u64);

        let blocks = stream::iter(0..num_blocks)
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
        let context = ha_ndarray::Context::default()?;
        let queue = ha_ndarray::Queue::new(context, self.block_size())?;
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
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
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
    type BlockWrite = ArrayBase<FileWriteGuardOwned<FE, Buffer<S::DType>>>;

    async fn write_block(&self, txn_id: TxnId, block_id: u64) -> TCResult<Self::BlockWrite> {
        let buffer = self.write_buffer(txn_id, block_id).await?;
        let block_axis = block_axis_for(self.shape(), self.block_size());
        let block_shape = block_shape_for(block_axis, self.shape(), buffer.len());
        ArrayBase::<FileWriteGuardOwned<FE, Buffer<S::DType>>>::new(block_shape, buffer)
            .map_err(TCError::from)
    }

    async fn write_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::BlockWrite>> {
        let num_blocks = div_ceil(self.size(), self.block_size() as u64);
        let blocks = stream::iter(0..num_blocks).then(move |block_id| {
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
    T: CDatatype,
    DenseAccess<Txn, FE, T>: From<S>,
{
    fn from(cow: DenseCow<FE, S>) -> Self {
        Self::Cow(Box::new(DenseCow {
            source: cow.source.into(),
            dir: cow.dir,
        }))
    }
}

impl<FE, S: fmt::Debug> fmt::Debug for DenseCow<FE, S> {
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
            let mut shape = source.shape().to_vec();
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
    ) -> TCResult<Vec<PermitRead<Range>>> {
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
impl<S: DenseInstance> DenseInstance for DenseExpand<S> {
    type Block = S::Block;
    type DType = S::DType;

    fn block_size(&self) -> usize {
        self.source.block_size()
    }

    async fn read_block(&self, txn_id: TxnId, block_id: u64) -> TCResult<Self::Block> {
        self.source.read_block(txn_id, block_id).await
    }

    async fn read_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::Block>> {
        self.source.read_blocks(txn_id).await
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> TCResult<Self::DType> {
        self.shape().validate_coord(&coord)?;
        let coord = self.transform.invert_coord(coord);
        self.source.read_value(txn_id, coord).await
    }
}

#[async_trait]
impl<S: TensorPermitRead + fmt::Debug> TensorPermitRead for DenseExpand<S> {
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        self.transform.shape().validate_range(&range)?;
        let range = self.transform.invert_range(range);
        self.source.read_permit(txn_id, range).await
    }
}

impl<Txn, FE, S, T> From<DenseExpand<S>> for DenseAccess<Txn, FE, T>
where
    S: Into<DenseAccess<Txn, FE, T>>,
    T: CDatatype,
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
pub struct DenseCombine<L, R, T: CDatatype> {
    left: L,
    right: R,
    block_op: Combine<T>,
    value_op: fn(T, T) -> T,
}

impl<L, R, T> DenseCombine<L, R, T>
where
    L: DenseInstance + fmt::Debug,
    R: DenseInstance + fmt::Debug,
    T: CDatatype + DType,
{
    pub fn new(left: L, right: R, block_op: Combine<T>, value_op: fn(T, T) -> T) -> TCResult<Self> {
        if left.block_size() == right.block_size() && left.shape() == right.shape() {
            Ok(Self {
                left,
                right,
                block_op,
                value_op,
            })
        } else {
            Err(bad_request!("cannot combine {:?} with {:?}", left, right))
        }
    }
}

impl<L, R, T> TensorInstance for DenseCombine<L, R, T>
where
    L: TensorInstance,
    R: TensorInstance,
    T: DType + CDatatype,
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
    T: CDatatype + DType,
{
    type Block = Array<T>;
    type DType = T;

    fn block_size(&self) -> usize {
        self.left.block_size()
    }

    async fn read_block(&self, txn_id: TxnId, block_id: u64) -> TCResult<Self::Block> {
        let (left, right) = try_join!(
            self.left.read_block(txn_id, block_id),
            self.right.read_block(txn_id, block_id)
        )?;

        (self.block_op)(left.into(), right.into())
    }

    async fn read_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::Block>> {
        let op = self.block_op;

        let (left, right) = try_join!(
            self.left.read_blocks(txn_id),
            self.right.read_blocks(txn_id)
        )?;

        let blocks = left.zip(right).map(move |(l, r)| {
            let l = l?;
            let r = r?;
            (op)(l.into(), r.into())
        });

        Ok(Box::pin(blocks))
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> TCResult<Self::DType> {
        let (left, right) = try_join!(
            self.left.read_value(txn_id, coord.to_vec()),
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
    T: CDatatype,
{
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
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
    T: CDatatype,
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
    T: CDatatype,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "combine {:?} with {:?}", self.left, self.right)
    }
}

#[derive(Clone)]
pub struct DenseCombineConst<S, T: CDatatype> {
    left: S,
    right: T,
    block_op: fn(Array<T>, T) -> TCResult<Array<T>>,
    value_op: fn(T, T) -> T,
}

impl<S: TensorInstance, T: CDatatype + DType> TensorInstance for DenseCombineConst<S, T> {
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
    T: CDatatype + DType,
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
impl<S: TensorPermitRead, T: CDatatype> TensorPermitRead for DenseCombineConst<S, T> {
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        self.left.read_permit(txn_id, range).await
    }
}

impl<Txn, FE, S, T> From<DenseCombineConst<S, T>> for DenseAccess<Txn, FE, T>
where
    S: Into<DenseAccess<Txn, FE, T>>,
    T: CDatatype,
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

impl<S: fmt::Debug, T: CDatatype> fmt::Debug for DenseCombineConst<S, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "combine {:?} with a constant value", self.left)
    }
}

type ArrayCmp<T> = fn(Block, Block) -> TCResult<Array<T>>;

pub struct DenseCompare<Txn, FE, T: CDatatype> {
    left: DenseAccessCast<Txn, FE>,
    right: DenseAccessCast<Txn, FE>,
    block_op: ArrayCmp<T>,
    value_op: fn(Number, Number) -> T,
}

impl<Txn, FE, T: CDatatype> Clone for DenseCompare<Txn, FE, T> {
    fn clone(&self) -> Self {
        Self {
            left: self.left.clone(),
            right: self.right.clone(),
            block_op: self.block_op,
            value_op: self.value_op,
        }
    }
}

impl<Txn, FE, T: CDatatype> DenseCompare<Txn, FE, T> {
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
            Err(bad_request!("cannot compare {:?} with {:?}", left, right))
        }
    }
}

impl<Txn, FE, T> TensorInstance for DenseCompare<Txn, FE, T>
where
    Txn: ThreadSafe,
    FE: ThreadSafe,
    T: CDatatype + DType,
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
    T: CDatatype + DType + fmt::Debug,
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
            self.left.read_value(txn_id, coord.to_vec()),
            self.right.read_value(txn_id, coord)
        )?;

        Ok((self.value_op)(left, right))
    }
}

#[async_trait]
impl<Txn: ThreadSafe, FE: ThreadSafe, T: CDatatype> TensorPermitRead for DenseCompare<Txn, FE, T> {
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        // always acquire these permits in-order to avoid the risk of a deadlock
        let mut left = self.left.read_permit(txn_id, range.clone()).await?;
        let right = self.right.read_permit(txn_id, range).await?;
        left.extend(right);
        Ok(left)
    }
}

impl<Txn, FE, T: CDatatype> From<DenseCompare<Txn, FE, T>> for DenseAccess<Txn, FE, T> {
    fn from(compare: DenseCompare<Txn, FE, T>) -> Self {
        Self::Compare(Box::new(compare))
    }
}

impl<Txn, FE, T: CDatatype> fmt::Debug for DenseCompare<Txn, FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "compare {:?} with {:?}", self.left, self.right)
    }
}

type ArrayCmpScalar<T> = fn(Block, Number) -> TCResult<Array<T>>;

pub struct DenseCompareConst<Txn, FE, T: CDatatype> {
    left: DenseAccessCast<Txn, FE>,
    right: Number,
    block_op: ArrayCmpScalar<T>,
    value_op: fn(Number, Number) -> T,
}

impl<Txn, FE, T: CDatatype> Clone for DenseCompareConst<Txn, FE, T> {
    fn clone(&self) -> Self {
        Self {
            left: self.left.clone(),
            right: self.right,
            block_op: self.block_op,
            value_op: self.value_op,
        }
    }
}

impl<Txn, FE, T: CDatatype> DenseCompareConst<Txn, FE, T> {
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
    T: CDatatype + DType,
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
    T: CDatatype + DType + fmt::Debug,
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
    T: CDatatype,
{
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        self.left.read_permit(txn_id, range).await
    }
}

impl<Txn, FE, T: CDatatype> From<DenseCompareConst<Txn, FE, T>> for DenseAccess<Txn, FE, T> {
    fn from(compare: DenseCompareConst<Txn, FE, T>) -> Self {
        Self::CompareConst(Box::new(compare))
    }
}

impl<Txn, FE, T: CDatatype> fmt::Debug for DenseCompareConst<Txn, FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "compare {:?} with {:?}", self.left, self.right)
    }
}

#[derive(Clone)]
pub struct DenseConst<L, T: CDatatype> {
    left: L,
    right: T,
    block_op: fn(Array<T>, T) -> TCResult<Array<T>>,
    value_op: fn(T, T) -> T,
}

impl<L, T: CDatatype> DenseConst<L, T> {
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
    T: CDatatype + DType,
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
    T: CDatatype + DType,
{
    type Block = Array<T>;
    type DType = T;

    fn block_size(&self) -> usize {
        self.left.block_size()
    }

    async fn read_block(&self, txn_id: TxnId, block_id: u64) -> TCResult<Self::Block> {
        self.left
            .read_block(txn_id, block_id)
            .map(move |result| result.and_then(|block| (self.block_op)(block.into(), self.right)))
            .await
    }

    async fn read_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::Block>> {
        let left = self.left.read_blocks(txn_id).await?;
        let blocks = left
            .map(move |result| result.and_then(|block| (self.block_op)(block.into(), self.right)));

        Ok(Box::pin(blocks))
    }

    async fn read_value(&self, txn_id: TxnId, coord: Coord) -> TCResult<Self::DType> {
        let left = self.left.read_value(txn_id, coord).await?;
        Ok((self.value_op)(left, self.right))
    }
}

#[async_trait]
impl<L: TensorPermitRead, T: CDatatype> TensorPermitRead for DenseConst<L, T> {
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        self.left.read_permit(txn_id, range).await
    }
}

impl<Txn, FE, L, T> From<DenseConst<L, T>> for DenseAccess<Txn, FE, T>
where
    L: Into<DenseAccess<Txn, FE, T>>,
    T: CDatatype,
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

impl<L: fmt::Debug, T: CDatatype> fmt::Debug for DenseConst<L, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "dual constant operation on {:?}", self.left)
    }
}

#[derive(Clone)]
pub struct DenseReduce<S, T: CDatatype> {
    source: S,
    transform: Reduce,
    block_map: ArrayBase<Arc<Vec<u64>>>,
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
        axes: Axes,
        keepdims: bool,
        id: S::DType,
        reduce_all: fn(Array<S::DType>, S::DType) -> TCResult<S::DType>,
        reduce_blocks: Combine<S::DType>,
        reduce_op: fn(Array<S::DType>, &[usize], bool) -> TCResult<Array<S::DType>>,
    ) -> TCResult<Self> {
        let num_blocks = div_ceil(source.size(), source.block_size() as u64);
        let block_axis = block_axis_for(source.shape(), source.block_size());
        let block_shape = block_shape_for(block_axis, source.shape(), source.block_size());
        debug_assert_eq!(source.shape()[block_axis] % block_shape[0] as u64, 0);

        let map_axes = axes[..(block_axis + 1)].to_vec();
        let block_axes = axes[block_axis..].to_vec();

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
        let block_map = ArrayBase::<Arc<Vec<u64>>>::new(block_map_shape, Arc::new(block_map))?;

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
                let block_max = block.max()?;
                let max = match NumberCollator::default().cmp(&max.into(), &block_max.into()) {
                    Ordering::Less => block_max,
                    Ordering::Equal | Ordering::Greater => max,
                };

                Ok(max)
            },
            |l, r| {
                let l = ArrayBase::<Arc<Buffer<S::DType>>>::copy(&l)?;
                let r = ArrayBase::<Arc<Buffer<S::DType>>>::copy(&r)?;

                l.clone()
                    .ge(r.clone())?
                    .cond(l, r)
                    .map(Array::from)
                    .map_err(TCError::from)
            },
            |mut block, axes, keepdims| {
                for x in axes.iter().rev().copied() {
                    block = block.max_axis(x, keepdims).map(Array::from)?
                }

                Ok(block)
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
                let block_min = block.min()?;
                let min = match NumberCollator::default().cmp(&min.into(), &block_min.into()) {
                    Ordering::Less | Ordering::Equal => min,
                    Ordering::Greater => block_min,
                };

                Ok(min)
            },
            |l, r| {
                let l = ArrayBase::<Arc<Buffer<S::DType>>>::copy(&l)?;
                let r = ArrayBase::<Arc<Buffer<S::DType>>>::copy(&r)?;

                l.clone()
                    .le(r.clone())?
                    .cond(l, r)
                    .map(Array::from)
                    .map_err(TCError::from)
            },
            |mut block, axes, keepdims| {
                for x in axes.iter().rev().copied() {
                    block = block.min_axis(x, keepdims).map(Array::from)?
                }

                Ok(block)
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
                    .sum()
                    .map(|block_sum| block_sum + sum)
                    .map_err(TCError::from)
            },
            |l, r| l.add(r).map(Array::from).map_err(TCError::from),
            |mut block, axes, keepdims| {
                for x in axes.iter().rev().copied() {
                    block = block.sum_axis(x, keepdims).map(Array::from)?
                }

                Ok(block)
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
                let block_product = block.product()?;
                Ok(block_product * product)
            },
            |l, r| l.mul(r).map(Array::from).map_err(TCError::from),
            |mut block, axes, keepdims| {
                for x in axes.iter().rev().copied() {
                    block = block.product_axis(x, keepdims).map(Array::from)?
                }

                Ok(block)
            },
        )
    }
}

impl<S: TensorInstance, T: CDatatype> TensorInstance for DenseReduce<S, T> {
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
    T: CDatatype + DType,
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
        );

        let mut map_slice = map_coord
            .into_iter()
            .map(|i| ha_ndarray::AxisBound::At(i))
            .collect::<Vec<_>>();

        for x in self.map_axes.iter().copied() {
            map_slice[x] = ha_ndarray::AxisBound::In(0, self.block_map.shape()[x], 1);
        }

        let context = ha_ndarray::Context::default()?;
        let queue = ha_ndarray::Queue::new(context, self.block_map.size())?;
        let block_map_slice = self.block_map.clone().slice(map_slice)?;
        let source_blocks = block_map_slice.read(&queue)?.to_slice()?;

        debug_assert_eq!(source_blocks.as_ref()[0], source_block_id);

        let block = self
            .source
            .read_block(txn_id, source_block_id)
            .map(|result| {
                result.and_then(|block| {
                    (self.reduce_op)(block, &self.block_axes, self.transform.keepdims())
                })
            })
            .await?;

        futures::stream::iter(source_blocks.as_ref().iter().skip(1).copied())
            .map(|source_block_id| {
                self.source
                    .read_block(txn_id, source_block_id)
                    .map(|result| {
                        result.and_then(|block| {
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
impl<S: TensorPermitRead, T: CDatatype> TensorPermitRead for DenseReduce<S, T> {
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        self.transform.shape().validate_range(&range)?;
        let range = self.transform.invert_range(range);
        self.source.read_permit(txn_id, range).await
    }
}

impl<Txn, FE, S, T> From<DenseReduce<S, T>> for DenseAccess<Txn, FE, T>
where
    S: Into<DenseAccess<Txn, FE, T>>,
    T: CDatatype,
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

impl<S: fmt::Debug, T: CDatatype> fmt::Debug for DenseReduce<S, T> {
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
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        if range.is_empty() || range == Range::all(self.transform.shape()) {
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
    T: CDatatype,
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
pub struct DenseSlice<S> {
    source: S,
    transform: Slice,
    block_map: ArrayBase<Vec<u64>>,
    block_size: usize,
}

impl<S: DenseInstance> DenseSlice<S> {
    pub fn new(source: S, range: Range) -> TCResult<Self> {
        let transform = Slice::new(source.shape().clone(), range)?;

        let block_axis = block_axis_for(source.shape(), source.block_size());
        let block_shape = block_shape_for(block_axis, source.shape(), source.block_size());
        let num_blocks = div_ceil(
            source.size() as u64,
            block_shape.iter().product::<usize>() as u64,
        ) as usize;

        let block_map_shape = source
            .shape()
            .iter()
            .take(block_axis)
            .copied()
            .map(|dim| {
                dim.try_into()
                    .map_err(|cause| bad_request!("invalid dimension: {dim} ({cause})"))
            })
            .collect::<Result<_, _>>()?;

        let block_map = ArrayBase::<Vec<_>>::new(
            block_map_shape,
            (0..num_blocks as u64).into_iter().collect(),
        )?;

        let mut block_map_bounds = Vec::with_capacity(block_axis + 1);
        for axis_range in transform.range().iter().take(block_axis).cloned() {
            let bound = axis_range.try_into()?;
            block_map_bounds.push(bound);
        }

        if transform.range().len() > block_axis {
            let bound = match &transform.range()[block_axis] {
                AxisRange::At(i) => {
                    let stride = block_map.shape().last().expect("stride");
                    let i = usize::try_from(*i)
                        .map_err(|cause| bad_request!("invalid index: {cause}"))?
                        / stride;
                    ha_ndarray::AxisBound::At(i)
                }
                AxisRange::In(axis_range, _step) => {
                    let stride = block_shape[0];
                    let start = usize::try_from(axis_range.start)
                        .map_err(|cause| bad_request!("invalid range start: {cause}"))?
                        / stride;
                    let stop = usize::try_from(axis_range.end)
                        .map_err(|cause| bad_request!("invalid range stop: {cause}"))?
                        / stride;
                    ha_ndarray::AxisBound::In(start, stop, 1)
                }
                AxisRange::Of(indices) => {
                    let stride = block_map.shape().last().expect("stride");
                    let indices = indices
                        .iter()
                        .copied()
                        .map(|i| {
                            usize::try_from(i)
                                .map(|i| i / stride)
                                .map_err(|cause| bad_request!("invalid index: {cause}"))
                        })
                        .collect::<Result<Vec<usize>, TCError>>()?;

                    ha_ndarray::AxisBound::Of(indices)
                }
            };

            block_map_bounds.push(bound);
        }

        let block_map = block_map.slice(block_map_bounds)?;
        let block_map = ArrayBase::<Vec<u64>>::copy(&block_map)?;

        let block_size = transform.shape().iter().product::<u64>() as usize / num_blocks;

        Ok(Self {
            source,
            transform,
            block_map,
            block_size,
        })
    }

    #[inline]
    fn block_bounds(&self, block_id: u64) -> TCResult<(u64, Vec<ha_ndarray::AxisBound>)> {
        let source_block_id = source_block_id_for(&self.block_map, block_id)?;

        let block_axis = block_axis_for(self.shape(), self.block_size);
        let block_shape = block_shape_for(block_axis, self.shape(), self.block_size);

        let local_bound =
            match ha_ndarray::AxisBound::try_from(self.transform.range()[block_axis].clone())? {
                ha_ndarray::AxisBound::At(i) => ha_ndarray::AxisBound::At(i),
                ha_ndarray::AxisBound::In(start, stop, step) => {
                    let stride = block_shape[0];

                    if source_block_id == 0 {
                        ha_ndarray::AxisBound::In(start, stride, step)
                    } else if source_block_id == self.block_map.size() as u64 - 1 {
                        ha_ndarray::AxisBound::In(stop - (stop % stride), stop, step)
                    } else {
                        let start = source_block_id as usize * stride;
                        ha_ndarray::AxisBound::In(start, start + stride, step)
                    }
                }
                ha_ndarray::AxisBound::Of(indices) => {
                    if source_block_id < indices.len() as u64 {
                        let i = indices[source_block_id as usize] as usize;
                        ha_ndarray::AxisBound::At(i)
                    } else {
                        return Err(bad_request!("block id {} is out of range", block_id));
                    }
                }
            };

        let mut block_bounds = Vec::with_capacity(self.ndim());
        for bound in self.transform.range().iter().take(block_axis).cloned() {
            block_bounds.push(bound.try_into()?);
        }

        if block_bounds.is_empty() {
            block_bounds.push(local_bound);
        } else {
            block_bounds[0] = local_bound;
        }

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
        let ndim = self.ndim();
        let transform = self.transform;
        let range = transform.range();
        let block_map = self.block_map;
        let source = self.source;

        let block_axis = block_axis_for(transform.shape(), self.block_size);
        let block_shape = block_shape_for(block_axis, transform.shape(), self.block_size);

        let local_bounds = match ha_ndarray::AxisBound::try_from(range[block_axis].clone())? {
            ha_ndarray::AxisBound::At(i) => {
                debug_assert_eq!(block_map.size(), 1);
                vec![ha_ndarray::AxisBound::At(i)]
            }
            ha_ndarray::AxisBound::In(start, stop, step) => {
                let stride = block_shape[0];

                if block_map.size() == 1 {
                    vec![ha_ndarray::AxisBound::In(start, stop, step)]
                } else {
                    let mut local_bounds = Vec::with_capacity(block_map.size());
                    local_bounds.push(ha_ndarray::AxisBound::In(start, stride, step));

                    for i in 0..(block_map.size() - 2) {
                        let start = stride * i;
                        local_bounds.push(ha_ndarray::AxisBound::In(start, start + stride, step));
                    }

                    local_bounds.push(ha_ndarray::AxisBound::In(
                        stop - (stop % stride),
                        stop,
                        step,
                    ));

                    local_bounds
                }
            }
            ha_ndarray::AxisBound::Of(indices) => {
                indices.into_iter().map(ha_ndarray::AxisBound::At).collect()
            }
        };

        let mut block_bounds = Vec::<ha_ndarray::AxisBound>::with_capacity(ndim);
        for bound in range.iter().skip(block_axis).cloned() {
            block_bounds.push(bound.try_into()?);
        }

        debug_assert_eq!(block_map.size(), local_bounds.len());
        let blocks = stream::iter(block_map.into_inner().into_iter().zip(local_bounds))
            .map(move |(block_id, local_bound)| {
                let mut block_bounds = block_bounds.to_vec();
                let source = source.clone();

                async move {
                    let block = get_block(source, block_id).await?;

                    if block_bounds.is_empty() {
                        block_bounds.push(local_bound);
                    } else {
                        block_bounds[0] = local_bound;
                    }

                    block.slice(block_bounds).map_err(TCError::from)
                }
            })
            .buffered(num_cpus::get());

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
        let (source_block_id, block_bounds) = self.block_bounds(block_id)?;
        let source_block = self.source.read_block(txn_id, source_block_id).await?;
        source_block.slice(block_bounds).map_err(TCError::from)
    }

    async fn read_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::Block>> {
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
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
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
    T: CDatatype,
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

impl<S: SparseInstance> From<S> for DenseSparse<S> {
    fn from(source: S) -> Self {
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
    type Block = ArrayBase<Vec<S::DType>>;
    type DType = S::DType;

    fn block_size(&self) -> usize {
        self.block_size
    }

    async fn read_block(&self, txn_id: TxnId, block_id: u64) -> TCResult<Self::Block> {
        let start = block_id * self.block_size() as u64;
        let stop = if (start + self.block_size() as u64) < self.size() {
            start + self.block_size() as u64
        } else {
            self.size()
        };

        let ndim = self.ndim();
        let strides = self
            .shape()
            .iter()
            .copied()
            .enumerate()
            .map(|(x, dim)| {
                if dim == 1 {
                    0
                } else {
                    self.shape().iter().rev().take(ndim - 1 - x).product()
                }
            })
            .collect::<Vec<u64>>();

        let start = coord_of(start, &strides, self.shape());
        let stop = coord_of(stop, &strides, self.shape());

        let range: Range = start
            .into_iter()
            .zip(stop)
            .map(|(from, to)| AxisRange::In(from..to, 1))
            .collect();

        let shape = range
            .shape()
            .into_vec()
            .into_iter()
            .map(|dim| dim as usize)
            .collect();

        let order = (0..self.ndim()).into_iter().collect();
        let elements = self.source.clone().elements(txn_id, range, order).await?;
        let values = ValueStream::new(elements, Range::all(self.source.shape()), S::DType::zero());
        let block = values.try_collect().await?;

        ArrayBase::<Vec<S::DType>>::new(shape, block).map_err(TCError::from)
    }

    async fn read_blocks(self, txn_id: TxnId) -> TCResult<BlockStream<Self::Block>> {
        let block_axis = block_axis_for(self.shape(), self.block_size());
        let block_shape = block_shape_for(block_axis, self.shape(), self.block_size());

        let range = Range::all(self.shape());
        let order = (0..self.ndim()).into_iter().collect();
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
                    ArrayBase::<Vec<_>>::new(block_shape, block).map_err(TCError::from)
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
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        self.source.read_permit(txn_id, range).await
    }
}

impl<Txn, FE, S, T> From<DenseSparse<S>> for DenseAccess<Txn, FE, T>
where
    S: Into<SparseAccess<Txn, FE, T>>,
    T: CDatatype,
{
    fn from(sparse: DenseSparse<S>) -> Self {
        Self::Sparse(DenseSparse {
            source: sparse.source.into(),
            block_size: sparse.block_size,
        })
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
    block_map: ArrayBase<Vec<u64>>,
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

        let permutation = transform.axes().to_vec();
        let (map_axes, block_axes) = permutation.split_at(block_axis);

        if map_axes.iter().copied().any(|x| x >= block_axis)
            || block_axes.iter().copied().any(|x| x <= block_axis)
        {
            return Err(bad_request!(
                "cannot transpose axes {:?} of {:?} without copying",
                block_axes,
                source
            ));
        }

        let block_map = ArrayBase::<Vec<_>>::new(map_shape, (0..num_blocks).into_iter().collect())?;
        let block_map = block_map.transpose(Some(map_axes.to_vec()))?;
        let block_map = ArrayBase::<Vec<_>>::copy(&block_map)?;

        Ok(Self {
            source,
            transform,
            block_map,
            block_axes: block_axes.to_vec(),
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

        let blocks = stream::iter(self.block_map.into_inner())
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
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        self.transform.shape().validate_range(&range)?;
        let range = self.transform.invert_range(&range);
        self.source.read_permit(txn_id, range).await
    }
}

impl<Txn, FE, S, T> From<DenseTranspose<S>> for DenseAccess<Txn, FE, T>
where
    T: CDatatype,
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
pub struct DenseUnary<S, T: CDatatype> {
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

impl<S: TensorInstance, T: CDatatype> TensorInstance for DenseUnary<S, T> {
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
impl<S: TensorPermitRead, T: CDatatype> TensorPermitRead for DenseUnary<S, T> {
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        self.source.read_permit(txn_id, range).await
    }
}

impl<Txn, FE, S, T> From<DenseUnary<S, T>> for DenseAccess<Txn, FE, T>
where
    S: Into<DenseAccess<Txn, FE, T>>,
    T: CDatatype,
{
    fn from(unary: DenseUnary<S, T>) -> Self {
        Self::Unary(Box::new(DenseUnary {
            source: unary.source.into(),
            block_op: unary.block_op,
            value_op: unary.value_op,
        }))
    }
}

impl<S: fmt::Debug, T: CDatatype> fmt::Debug for DenseUnary<S, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "unary transform of {:?}", self.source)
    }
}

pub struct DenseUnaryCast<Txn, FE, T: CDatatype> {
    source: DenseAccessCast<Txn, FE>,
    block_op: fn(Block) -> TCResult<Array<T>>,
    value_op: fn(Number) -> T,
}

impl<Txn, FE, T: CDatatype> Clone for DenseUnaryCast<Txn, FE, T> {
    fn clone(&self) -> Self {
        Self {
            source: self.source.clone(),
            block_op: self.block_op,
            value_op: self.value_op,
        }
    }
}

impl<Txn, FE, T: CDatatype> DenseUnaryCast<Txn, FE, T> {
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
    T: CDatatype + DType,
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
    T: CDatatype + DType + fmt::Debug,
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
    T: CDatatype,
{
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<Vec<PermitRead<Range>>> {
        let source = &self.source;
        cast_dispatch!(source, this, this.read_permit(txn_id, range).await)
    }
}

impl<Txn, FE, T: CDatatype> From<DenseUnaryCast<Txn, FE, T>> for DenseAccess<Txn, FE, T> {
    fn from(unary: DenseUnaryCast<Txn, FE, T>) -> Self {
        Self::UnaryCast(Box::new(unary))
    }
}

impl<Txn, FE, T: CDatatype> fmt::Debug for DenseUnaryCast<Txn, FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "unary transform/cast of {:?}", self.source)
    }
}

#[inline]
fn block_axis_for(shape: &[u64], block_size: usize) -> usize {
    debug_assert!(!shape.is_empty());
    debug_assert!(shape.iter().copied().all(|dim| dim > 0));
    debug_assert!(shape.iter().product::<u64>() >= block_size as u64);

    let mut block_ndim = 1;
    let mut size = 1;
    for dim in shape.iter().rev() {
        size *= dim;

        if size > block_size as u64 {
            break;
        } else {
            block_ndim += 1;
        }
    }

    shape.len() - block_ndim
}

#[inline]
fn block_shape_for(axis: usize, shape: &[u64], block_size: usize) -> BlockShape {
    if axis == shape.len() - 1 {
        vec![block_size]
    } else {
        let axis_dim = (shape.iter().skip(axis).product::<u64>() / block_size as u64) as usize;
        debug_assert_eq!(block_size % axis_dim, 0);

        let mut block_shape = BlockShape::with_capacity(shape.len() - axis + 1);
        block_shape.push(axis_dim);
        block_shape.extend(shape.iter().skip(axis).copied().map(|dim| dim as usize));

        debug_assert!(!block_shape.is_empty());

        block_shape
    }
}

#[inline]
fn source_block_id_for(block_map: &ArrayBase<Vec<u64>>, block_id: u64) -> TCResult<u64> {
    block_map
        .as_slice()
        .get(block_id as usize)
        .copied()
        .ok_or_else(|| bad_request!("block id {} is out of range", block_id))
}
