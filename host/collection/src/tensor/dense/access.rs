use std::marker::PhantomData;
use std::sync::Arc;
use std::{fmt, io};

use async_trait::async_trait;
use destream::de;
use freqfs::{
    DirLock, DirReadGuard, FileLoad, FileReadGuard, FileReadGuardOwned, FileWriteGuardOwned,
};
use futures::future::{Future, FutureExt, TryFutureExt};
use futures::stream::{self, Stream, StreamExt, TryStreamExt};
use ha_ndarray::*;
use safecast::AsType;

use tc_error::*;
use tc_transact::lock::{PermitRead, PermitWrite};
use tc_transact::TxnId;
use tc_value::{DType, NumberClass, NumberInstance, NumberType};

use super::{DenseInstance, DenseWrite, DenseWriteGuard, DenseWriteLock};

use crate::tensor::transform::{Broadcast, Expand, Reshape, Slice, Transpose};
use crate::tensor::{
    offset_of, AccessPermit, Axes, AxisRange, Coord, Range, Semaphore, Shape, TensorInstance,
    ERR_READ_ONLY, IDEAL_BLOCK_SIZE,
};

use super::stream::BlockResize;
use super::{BlockShape, BlockStream, DenseCacheFile};

pub enum DenseAccess<FE, T> {
    File(DenseFile<FE, T>),
    Broadcast(Box<DenseBroadcast<Self>>),
    Cast(Box<DenseCast<FE, T>>),
    Cow(Box<DenseCow<FE, Self>>),
    Expand(Box<DenseExpand<Self>>),
    Reshape(Box<DenseReshape<Self>>),
    Slice(Box<DenseSlice<Self>>),
    Transpose(Box<DenseTranspose<Self>>),
    Version(DenseVersion<FE, T>),
}

impl<FE, T> Clone for DenseAccess<FE, T> {
    fn clone(&self) -> Self {
        match self {
            Self::File(file) => Self::File(file.clone()),
            Self::Broadcast(broadcast) => Self::Broadcast(broadcast.clone()),
            Self::Cast(cast) => Self::Cast(cast.clone()),
            Self::Cow(cow) => Self::Cow(cow.clone()),
            Self::Expand(expand) => Self::Expand(expand.clone()),
            Self::Reshape(reshape) => Self::Reshape(reshape.clone()),
            Self::Slice(slice) => Self::Slice(slice.clone()),
            Self::Transpose(transpose) => Self::Transpose(transpose.clone()),
            Self::Version(version) => Self::Version(version.clone()),
        }
    }
}

macro_rules! access_dispatch {
    ($this:ident, $var:ident, $call:expr) => {
        match $this {
            Self::File($var) => $call,
            Self::Broadcast($var) => $call,
            Self::Cast($var) => $call,
            Self::Cow($var) => $call,
            Self::Expand($var) => $call,
            Self::Reshape($var) => $call,
            Self::Slice($var) => $call,
            Self::Transpose($var) => $call,
            Self::Version($var) => $call,
        }
    };
}

impl<FE, T> TensorInstance for DenseAccess<FE, T>
where
    FE: Send + Sync + 'static,
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
impl<FE, T> DenseInstance for DenseAccess<FE, T>
where
    FE: DenseCacheFile + AsType<Buffer<T>>,
    T: CDatatype + DType,
    Buffer<T>: de::FromStream<Context = ()>,
{
    type Block = Array<T>;
    type DType = T;

    fn block_size(&self) -> usize {
        access_dispatch!(self, this, this.block_size())
    }

    async fn read_block(&self, block_id: u64) -> TCResult<Self::Block> {
        access_dispatch!(
            self,
            this,
            this.read_block(block_id).map_ok(Array::from).await
        )
    }

    async fn read_blocks(self) -> TCResult<BlockStream<Self::Block>> {
        match self {
            Self::Cast(cast) => Ok(Box::pin(cast.read_blocks().await?)),
            Self::File(file) => Ok(Box::pin(file.read_blocks().await?.map_ok(Array::from))),
            Self::Broadcast(broadcast) => {
                Ok(Box::pin(broadcast.read_blocks().await?.map_ok(Array::from)))
            }
            Self::Cow(cow) => Ok(Box::pin(cow.read_blocks().await?.map_ok(Array::from))),
            Self::Expand(expand) => Ok(Box::pin(expand.read_blocks().await?.map_ok(Array::from))),
            Self::Reshape(reshape) => {
                Ok(Box::pin(reshape.read_blocks().await?.map_ok(Array::from)))
            }
            Self::Slice(slice) => Ok(Box::pin(slice.read_blocks().await?.map_ok(Array::from))),
            Self::Transpose(transpose) => {
                Ok(Box::pin(transpose.read_blocks().await?.map_ok(Array::from)))
            }
            Self::Version(version) => {
                Ok(Box::pin(version.read_blocks().await?.map_ok(Array::from)))
            }
        }
    }
}

#[async_trait]
impl<FE, T> AccessPermit for DenseAccess<FE, T>
where
    FE: Send + Sync,
    T: CDatatype + DType,
{
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<PermitRead<Range>> {
        match self {
            Self::Broadcast(broadcast) => broadcast.read_permit(txn_id, range).await,
            Self::Cast(cast) => cast.read_permit(txn_id, range).await,
            Self::Cow(cow) => cow.read_permit(txn_id, range).await,
            Self::Expand(expand) => expand.read_permit(txn_id, range).await,
            Self::Reshape(reshape) => reshape.read_permit(txn_id, range).await,
            Self::Slice(slice) => slice.read_permit(txn_id, range).await,
            Self::Transpose(transpose) => transpose.read_permit(txn_id, range).await,
            Self::Version(version) => version.read_permit(txn_id, range).await,

            other => Err(bad_request!(
                "{:?} does not support transactional locking",
                other
            )),
        }
    }

    async fn write_permit(&self, txn_id: TxnId, range: Range) -> TCResult<PermitWrite<Range>> {
        match self {
            Self::Slice(slice) => slice.write_permit(txn_id, range).await,
            Self::Version(version) => version.write_permit(txn_id, range).await,
            other => Err(bad_request!(
                "{:?} does not support transactional writes",
                other
            )),
        }
    }
}

impl<FE, T: DType> fmt::Debug for DenseAccess<FE, T> {
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

        let ndim = shape.len();
        let size = shape.iter().product();

        let ideal_block_size = IDEAL_BLOCK_SIZE as u64;
        let (block_size, num_blocks) = if size < (2 * ideal_block_size) {
            (size as usize, 1)
        } else if ndim == 1 && size % ideal_block_size == 0 {
            (IDEAL_BLOCK_SIZE, (size / ideal_block_size) as usize)
        } else if ndim == 1
            || (shape.iter().rev().take(2).product::<u64>() > (2 * ideal_block_size))
        {
            let num_blocks = div_ceil(size, ideal_block_size) as usize;
            (IDEAL_BLOCK_SIZE, num_blocks as usize)
        } else {
            let matrix_size = shape.iter().rev().take(2).product::<u64>();
            let block_size = ideal_block_size + (matrix_size - (ideal_block_size % matrix_size));
            let num_blocks = div_ceil(size, ideal_block_size);
            (block_size as usize, num_blocks as usize)
        };

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

        let block_map =
            ArrayBase::<Vec<_>>::new(map_shape, (0u64..num_blocks as u64).into_iter().collect())?;

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
    FE: Send + Sync + 'static,
    T: DType + Send + Sync + 'static,
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

    async fn read_block(&self, block_id: u64) -> TCResult<Self::Block> {
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

    async fn read_blocks(self) -> TCResult<BlockStream<Self::Block>> {
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
}

#[async_trait]
impl<'a, FE, T> DenseWrite for DenseFile<FE, T>
where
    FE: FileLoad + AsType<Buffer<T>>,
    T: CDatatype + DType + 'static,
    Buffer<T>: de::FromStream<Context = ()>,
{
    type BlockWrite = ArrayBase<FileWriteGuardOwned<FE, Buffer<T>>>;

    async fn write_block(&self, block_id: u64) -> TCResult<Self::BlockWrite> {
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

    async fn write_blocks(self) -> TCResult<BlockStream<Self::BlockWrite>> {
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
    T: CDatatype + DType + 'static,
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

impl<FE, T> From<DenseFile<FE, T>> for DenseAccess<FE, T> {
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
    async fn overwrite<O: DenseInstance<DType = T>>(&self, other: O) -> TCResult<()> {
        let block_axis = block_axis_for(&self.shape, self.block_size);
        let block_shape = block_shape_for(block_axis, &self.shape, self.block_size);

        let context = ha_ndarray::Context::default()?;
        let queue = ha_ndarray::Queue::new(context, block_shape.iter().product())?;

        let blocks = other.read_blocks().await?;
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

    async fn overwrite_value(&self, value: T) -> TCResult<()> {
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

    async fn write_value(&self, coord: Coord, value: T) -> TCResult<()> {
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
    DenseFile<FE, T>: TensorInstance,
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
    FE: Send + Sync,
    T: CDatatype + DType,
    DenseFile<FE, T>: DenseInstance,
{
    type Block = <DenseFile<FE, T> as DenseInstance>::Block;
    type DType = <DenseFile<FE, T> as DenseInstance>::DType;

    fn block_size(&self) -> usize {
        self.file.block_size()
    }

    async fn read_block(&self, block_id: u64) -> TCResult<Self::Block> {
        self.file.read_block(block_id).await
    }

    async fn read_blocks(self) -> TCResult<BlockStream<Self::Block>> {
        self.file.read_blocks().await
    }
}

#[async_trait]
impl<FE, T> AccessPermit for DenseVersion<FE, T>
where
    FE: Send + Sync,
    T: CDatatype + DType,
{
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<PermitRead<Range>> {
        self.semaphore
            .read(txn_id, range)
            .map_err(TCError::from)
            .await
    }

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
    FE: Send + Sync,
    T: CDatatype + DType,
    DenseFile<FE, T>: DenseWriteLock<'a>,
{
    type WriteGuard = <DenseFile<FE, T> as DenseWriteLock<'a>>::WriteGuard;

    async fn write(&'a self) -> Self::WriteGuard {
        self.file.write().await
    }
}

impl<FE, T> From<DenseVersion<FE, T>> for DenseAccess<FE, T> {
    fn from(version: DenseVersion<FE, T>) -> Self {
        Self::Version(version)
    }
}

impl<FE, T> fmt::Debug for DenseVersion<FE, T>
where
    DenseFile<FE, T>: fmt::Debug,
{
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

    async fn read_block(&self, block_id: u64) -> TCResult<Self::Block> {
        let source_block_id = source_block_id_for(&self.block_map, block_id)?;
        let block_axis = block_axis_for(self.shape(), self.block_size);
        let block_shape = block_shape_for(block_axis, self.shape(), self.block_size);
        let source_block = self.source.read_block(source_block_id).await?;
        source_block.broadcast(block_shape).map_err(TCError::from)
    }

    async fn read_blocks(self) -> TCResult<BlockStream<Self::Block>> {
        let block_axis = block_axis_for(self.shape(), self.block_size);
        let block_shape = block_shape_for(block_axis, self.shape(), self.block_size);

        let blocks = stream::iter(self.block_map.into_inner())
            .map(move |block_id| {
                let source = self.source.clone();
                async move { source.read_block(block_id).await }
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
}

#[async_trait]
impl<S: AccessPermit> AccessPermit for DenseBroadcast<S> {
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<PermitRead<Range>> {
        self.transform.shape().validate_range(&range)?;
        let range = self.transform.invert_range(range);
        self.source.read_permit(txn_id, range).await
    }

    async fn write_permit(&self, txn_id: TxnId, range: Range) -> TCResult<PermitWrite<Range>> {
        self.transform.shape().validate_range(&range)?;
        let range = self.transform.invert_range(range);
        self.source.write_permit(txn_id, range).await
    }
}

impl<FE, T, S: Into<DenseAccess<FE, T>>> From<DenseBroadcast<S>> for DenseAccess<FE, T> {
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

pub enum DenseCastSource<FE> {
    F32(DenseAccess<FE, f32>),
    F64(DenseAccess<FE, f64>),
    I16(DenseAccess<FE, i16>),
    I32(DenseAccess<FE, i32>),
    I64(DenseAccess<FE, i64>),
    U8(DenseAccess<FE, u8>),
    U16(DenseAccess<FE, u16>),
    U32(DenseAccess<FE, u32>),
    U64(DenseAccess<FE, u64>),
}

impl<FE> Clone for DenseCastSource<FE> {
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

macro_rules! cast_source {
    ($t:ty, $var:ident) => {
        impl<FE> From<DenseAccess<FE, $t>> for DenseCastSource<FE> {
            fn from(access: DenseAccess<FE, $t>) -> Self {
                Self::$var(access)
            }
        }
    };
}

cast_source!(f32, F32);
cast_source!(f64, F64);
cast_source!(i16, I16);
cast_source!(i32, I32);
cast_source!(i64, I64);
cast_source!(u8, U8);
cast_source!(u16, U16);
cast_source!(u32, U32);
cast_source!(u64, U64);

macro_rules! cast_dispatch {
    ($this:ident, $var:ident, $call:expr) => {
        match $this {
            DenseCastSource::F32($var) => $call,
            DenseCastSource::F64($var) => $call,
            DenseCastSource::I16($var) => $call,
            DenseCastSource::I32($var) => $call,
            DenseCastSource::I64($var) => $call,
            DenseCastSource::U8($var) => $call,
            DenseCastSource::U16($var) => $call,
            DenseCastSource::U32($var) => $call,
            DenseCastSource::U64($var) => $call,
        }
    };
}

impl<FE> fmt::Debug for DenseCastSource<FE> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        cast_dispatch!(self, this, this.fmt(f))
    }
}

pub struct DenseCast<FE, T> {
    source: DenseCastSource<FE>,
    dtype: PhantomData<T>,
}

impl<FE, T> Clone for DenseCast<FE, T> {
    fn clone(&self) -> Self {
        Self {
            source: self.source.clone(),
            dtype: self.dtype,
        }
    }
}

impl<FE, T> DenseCast<FE, T> {
    pub fn new<S: Into<DenseCastSource<FE>>>(source: S) -> Self {
        Self {
            source: source.into(),
            dtype: PhantomData,
        }
    }
}

impl<FE: Send + Sync + 'static, T: CDatatype + DType> TensorInstance for DenseCast<FE, T> {
    fn dtype(&self) -> NumberType {
        T::dtype()
    }

    fn shape(&self) -> &Shape {
        let source = &self.source;
        cast_dispatch!(source, this, this.shape())
    }
}

#[async_trait]
impl<FE, T> DenseInstance for DenseCast<FE, T>
where
    FE: DenseCacheFile + AsType<Buffer<T>>,
    T: CDatatype + DType,
    Buffer<T>: de::FromStream<Context = ()>,
{
    type Block = Array<T>;
    type DType = T;

    fn block_size(&self) -> usize {
        let source = &self.source;
        cast_dispatch!(source, this, this.block_size())
    }

    async fn read_block(&self, block_id: u64) -> TCResult<Self::Block> {
        let source = &self.source;

        cast_dispatch!(
            source,
            this,
            this.read_block(block_id)
                .map(|result| result
                    .and_then(|block| block.cast().map(Array::from).map_err(TCError::from)))
                .await
        )
    }

    async fn read_blocks(self) -> TCResult<BlockStream<Self::Block>> {
        let source = self.source;

        cast_dispatch!(source, this, {
            let source_blocks = this.read_blocks().await?;
            let blocks = source_blocks.map(|result| {
                result.and_then(|block| block.cast().map(Array::from).map_err(TCError::from))
            });

            Ok(Box::pin(blocks))
        })
    }
}

#[async_trait]
impl<FE, T> AccessPermit for DenseCast<FE, T>
where
    FE: Send + Sync,
    T: CDatatype + DType,
    DenseAccess<FE, T>: AccessPermit,
{
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<PermitRead<Range>> {
        let source = &self.source;
        cast_dispatch!(source, this, this.read_permit(txn_id, range).await)
    }

    async fn write_permit(&self, txn_id: TxnId, range: Range) -> TCResult<PermitWrite<Range>> {
        let source = &self.source;
        cast_dispatch!(source, this, this.write_permit(txn_id, range).await)
    }
}

impl<FE, T: DType> fmt::Debug for DenseCast<FE, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "cast {:?} into {:?}", self.source, T::dtype())
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
        block_id: u64,
    ) -> TCResult<FileWriteGuardOwned<FE, Buffer<S::DType>>> {
        let mut dir = self.dir.write().await;

        if let Some(buffer) = dir.get_file(&block_id) {
            buffer.write_owned().map_err(TCError::from).await
        } else {
            let block = self.source.read_block(block_id).await?;

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
    FE: Send + Sync + 'static,
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

    async fn read_block(&self, block_id: u64) -> TCResult<Self::Block> {
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
            self.source.read_block(block_id).map_ok(Array::from).await
        }
    }

    async fn read_blocks(self) -> TCResult<BlockStream<Self::Block>> {
        let num_blocks = div_ceil(self.size(), self.block_size() as u64);

        let blocks = stream::iter(0..num_blocks)
            .map(move |block_id| {
                let this = self.clone();
                async move { this.read_block(block_id).await }
            })
            .buffered(num_cpus::get());

        Ok(Box::pin(blocks))
    }
}

#[async_trait]
impl<FE, S> AccessPermit for DenseCow<FE, S>
where
    FE: Send + Sync,
    S: AccessPermit,
{
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<PermitRead<Range>> {
        self.source.read_permit(txn_id, range).await
    }

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

    async fn write_block(&self, block_id: u64) -> TCResult<Self::BlockWrite> {
        let buffer = self.write_buffer(block_id).await?;
        let block_axis = block_axis_for(self.shape(), self.block_size());
        let block_shape = block_shape_for(block_axis, self.shape(), buffer.len());
        ArrayBase::<FileWriteGuardOwned<FE, Buffer<S::DType>>>::new(block_shape, buffer)
            .map_err(TCError::from)
    }

    async fn write_blocks(self) -> TCResult<BlockStream<Self::BlockWrite>> {
        let num_blocks = div_ceil(self.size(), self.block_size() as u64);
        let blocks = stream::iter(0..num_blocks).then(move |block_id| {
            let this = self.clone();
            async move { this.write_block(block_id).await }
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

impl<'a, FE, S, T> From<DenseCow<FE, S>> for DenseAccess<FE, T>
where
    DenseAccess<FE, T>: From<S>,
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
    async fn overwrite<O: DenseInstance<DType = S::DType>>(&self, other: O) -> TCResult<()> {
        let source = other.read_blocks().await?;

        let block_axis = block_axis_for(self.cow.shape(), self.cow.block_size());
        let block_shape = block_shape_for(block_axis, self.cow.shape(), self.cow.block_size());
        let source = BlockResize::new(source, block_shape)?;

        let dest = self.cow.clone().write_blocks().await?;

        dest.zip(source)
            .map(|(dest, source)| {
                let mut dest = dest?;
                let source = source?;
                dest.write(&source).map_err(TCError::from)
            })
            .try_fold((), |(), _| futures::future::ready(Ok(())))
            .await
    }

    async fn overwrite_value(&self, value: S::DType) -> TCResult<()> {
        let dest = self.cow.clone().write_blocks().await?;
        dest.map_ok(|mut block| block.write_value(value))
            .try_fold((), |(), _| futures::future::ready(Ok(())))
            .await
    }

    async fn write_value(&self, coord: Coord, value: S::DType) -> TCResult<()> {
        self.cow.shape().validate_coord(&coord)?;

        let offset = offset_of(coord, self.cow.shape());
        let block_id = offset / self.cow.block_size() as u64;
        let block_offset = offset % self.cow.block_size() as u64;
        let mut buffer = self.cow.write_buffer(block_id).await?;

        buffer
            .write_value_at(block_offset as usize, value)
            .map_err(TCError::from)
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

    async fn read_block(&self, block_id: u64) -> TCResult<Self::Block> {
        self.source.read_block(block_id).await
    }

    async fn read_blocks(self) -> TCResult<BlockStream<Self::Block>> {
        self.source.read_blocks().await
    }
}

#[async_trait]
impl<S: AccessPermit + fmt::Debug> AccessPermit for DenseExpand<S> {
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<PermitRead<Range>> {
        self.transform.shape().validate_range(&range)?;
        let range = self.transform.invert_range(range);
        self.source.read_permit(txn_id, range).await
    }

    async fn write_permit(&self, _txn_id: TxnId, _range: Range) -> TCResult<PermitWrite<Range>> {
        Err(bad_request!("{:?} is {}", self, ERR_READ_ONLY))
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

    async fn read_block(&self, block_id: u64) -> TCResult<Self::Block> {
        let block_axis = block_axis_for(self.shape(), self.block_size());
        let mut block_shape = block_shape_for(block_axis, self.shape(), self.block_size());

        let block = self.source.read_block(block_id).await?;

        if block.size() < self.block_size() {
            // this must be the trailing block
            let axis_dim = self.block_size() / block_shape.iter().skip(1).product::<usize>();
            block_shape[0] = axis_dim;
        }

        block.reshape(block_shape).map_err(TCError::from)
    }

    async fn read_blocks(self) -> TCResult<BlockStream<Self::Block>> {
        let block_size = self.block_size();
        let block_axis = block_axis_for(self.shape(), block_size);
        let block_shape = block_shape_for(block_axis, self.shape(), block_size);

        let source_blocks = self.source.read_blocks().await?;
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
}

#[async_trait]
impl<S: AccessPermit + fmt::Debug> AccessPermit for DenseReshape<S> {
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<PermitRead<Range>> {
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

    async fn write_permit(&self, _txn_id: TxnId, _range: Range) -> TCResult<PermitWrite<Range>> {
        Err(bad_request!("{:?} is {}", self, ERR_READ_ONLY))
    }
}

impl<FE, T, S: Into<DenseAccess<FE, T>>> From<DenseReshape<S>> for DenseAccess<FE, T> {
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

    async fn read_block(&self, block_id: u64) -> TCResult<Self::Block> {
        let (source_block_id, block_bounds) = self.block_bounds(block_id)?;
        let source_block = self.source.read_block(source_block_id).await?;
        source_block.slice(block_bounds).map_err(TCError::from)
    }

    async fn read_blocks(self) -> TCResult<BlockStream<Self::Block>> {
        let blocks = self
            .block_stream(|source, block_id| async move { source.read_block(block_id).await })
            .await?;

        Ok(Box::pin(blocks))
    }
}

#[async_trait]
impl<S: AccessPermit> AccessPermit for DenseSlice<S> {
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<PermitRead<Range>> {
        self.transform.shape().validate_range(&range)?;
        let range = self.transform.invert_range(range);
        self.source.read_permit(txn_id, range).await
    }

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

    async fn write_block(&self, block_id: u64) -> TCResult<Self::BlockWrite> {
        let (source_block_id, block_bounds) = self.block_bounds(block_id)?;
        let source_block = self.source.write_block(source_block_id).await?;
        source_block.slice(block_bounds).map_err(TCError::from)
    }

    async fn write_blocks(self) -> TCResult<BlockStream<Self::BlockWrite>> {
        let blocks = self
            .block_stream(|source, block_id| async move { source.write_block(block_id).await })
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

impl<FE, T, S: Into<DenseAccess<FE, T>>> From<DenseSlice<S>> for DenseAccess<FE, T> {
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
    async fn overwrite<O: DenseInstance<DType = S::DType>>(&self, other: O) -> TCResult<()> {
        let block_axis = block_axis_for(self.dest.shape(), self.dest.block_size);
        let block_shape = block_shape_for(block_axis, self.dest.shape(), self.dest.block_size);

        let dest = self.dest.clone().write_blocks().await?;
        let source = other.read_blocks().await?;
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

    async fn overwrite_value(&self, value: S::DType) -> TCResult<()> {
        let dest = self.dest.clone().write_blocks().await?;
        dest.map_ok(|mut block| block.write_value(value))
            .try_fold((), |(), _| futures::future::ready(Ok(())))
            .await
    }

    async fn write_value(&self, coord: Coord, value: S::DType) -> TCResult<()> {
        let source_coord = self.dest.transform.invert_coord(coord)?;
        let source = self.dest.source.write().await;
        source.write_value(source_coord, value).await
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

    async fn read_block(&self, block_id: u64) -> TCResult<Self::Block> {
        let source_block_id = source_block_id_for(&self.block_map, block_id)?;
        let block = self.source.read_block(source_block_id).await?;

        block
            .transpose(Some(self.transform.axes().to_vec()))
            .map_err(TCError::from)
    }

    async fn read_blocks(self) -> TCResult<BlockStream<Self::Block>> {
        let block_axes = self.block_axes;

        let blocks = stream::iter(self.block_map.into_inner())
            .map(move |block_id| {
                let source = self.source.clone();
                async move { source.read_block(block_id).await }
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
}

#[async_trait]
impl<S: AccessPermit> AccessPermit for DenseTranspose<S> {
    async fn read_permit(&self, txn_id: TxnId, range: Range) -> TCResult<PermitRead<Range>> {
        self.transform.shape().validate_range(&range)?;
        let range = self.transform.invert_range(&range);
        self.source.read_permit(txn_id, range).await
    }

    async fn write_permit(&self, txn_id: TxnId, range: Range) -> TCResult<PermitWrite<Range>> {
        self.transform.shape().validate_range(&range)?;
        let range = self.transform.invert_range(&range);
        self.source.write_permit(txn_id, range).await
    }
}

impl<FE, T, S: Into<DenseAccess<FE, T>>> From<DenseTranspose<S>> for DenseAccess<FE, T> {
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
fn div_ceil(num: u64, denom: u64) -> u64 {
    if num % denom == 0 {
        num / denom
    } else {
        (num / denom) + 1
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
