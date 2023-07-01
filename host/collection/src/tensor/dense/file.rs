use std::marker::PhantomData;
use std::sync::Arc;
use std::{fmt, io};

use async_trait::async_trait;
use destream::de;
use freqfs::*;
use futures::stream::{StreamExt, TryStreamExt};
use ha_ndarray::{
    ArrayBase, ArrayOp, Buffer, BufferWrite, CDatatype, NDArray, NDArrayMathScalar, NDArrayRead,
};
use itertools::Itertools;
use safecast::{AsType, CastFrom};

use tc_error::*;
use tc_transact::TxnId;
use tc_value::{DType, Number, NumberClass, NumberType};
use tcgeneric::ThreadSafe;

use crate::tensor::{offset_of, Coord, Shape, TensorInstance};

use super::access::DenseAccess;
use super::stream::BlockResize;
use super::{
    block_axis_for, block_map_for, block_shape_for, div_ceil, ideal_block_size_for, BlockStream,
    DenseCacheFile, DenseInstance, DenseWrite, DenseWriteGuard, DenseWriteLock,
};

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
    T: CDatatype + DType,
    Buffer<T>: de::FromStream<Context = ()>,
{
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

    pub async fn copy_from<O>(dir: DirLock<FE>, txn_id: TxnId, other: O) -> TCResult<Self>
    where
        O: DenseInstance<DType = T>,
    {
        let shape = other.shape().clone();
        let (block_size, num_blocks) = ideal_block_size_for(&shape);
        let block_axis = block_axis_for(&shape, block_size);
        let block_shape = block_shape_for(block_axis, &shape, block_size);
        let block_map = block_map_for(num_blocks as u64, shape.as_slice(), &block_shape)?;

        let source_blocks = other.read_blocks(txn_id).await?;
        let mut source_blocks = BlockResize::new(source_blocks, block_shape)?;

        let mut dir_guard = dir.write().await;

        let mut block_id = 0;
        while let Some(block) = source_blocks.try_next().await? {
            let buffer = Buffer::from(block.into_inner());
            let size_in_bytes = buffer.len() * T::dtype().size();
            dir_guard.create_file(block_id.to_string(), buffer, size_in_bytes)?;
            block_id += 1;
        }

        if block_id != num_blocks - 1 {
            return Err(bad_request!("cannot create a tensor of shape {shape:?} from {num_blocks} blocks of size {block_size}"));
        }

        std::mem::drop(dir_guard);

        Ok(Self {
            dir,
            block_map,
            block_size,
            shape,
            dtype: PhantomData,
        })
    }

    pub async fn from_values(dir: DirLock<FE>, shape: Shape, values: Vec<Number>) -> TCResult<Self>
    where
        T: CastFrom<Number>,
    {
        shape.validate()?;

        if values.len() as u64 != shape.size() {
            return Err(bad_request!(
                "cannot construct a tensor of shape {shape:?} from {len} values",
                len = values.len()
            ));
        }

        let (block_size, num_blocks) = ideal_block_size_for(&shape);
        let block_axis = block_axis_for(&shape, block_size);
        let block_shape = block_shape_for(block_axis, &shape, block_size);
        let block_map = block_map_for(num_blocks as u64, shape.as_slice(), &block_shape)?;

        let blocks = values
            .into_iter()
            .map(|n| T::cast_from(n))
            .chunks(block_size);

        let mut dir_guard = dir.write().await;

        for (block_id, block) in blocks.into_iter().enumerate() {
            let buffer = Buffer::from(block.collect::<Vec<T>>());
            let size_in_bytes = buffer.len() * T::dtype().size();
            dir_guard.create_file(block_id.to_string(), buffer, size_in_bytes)?;
        }

        std::mem::drop(dir_guard);

        Ok(Self {
            dir,
            block_map,
            block_size,
            shape,
            dtype: PhantomData,
        })
    }

    pub async fn load(dir: DirLock<FE>, shape: Shape) -> TCResult<Self> {
        let contents = dir.write().await;
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

        let block_map = block_map_for(num_blocks as u64, shape.as_slice(), &block_shape)?;

        Ok(Self {
            dir,
            block_map,
            block_size,
            shape,
            dtype: PhantomData,
        })
    }

    pub async fn range(dir: DirLock<FE>, shape: Shape, start: T, stop: T) -> TCResult<Self>
    where
        T: fmt::Display,
    {
        Self::construct_with_op(dir, shape, |context, queue, block_size| {
            let op = ha_ndarray::construct::Range::with_context(context, start, stop, block_size)?;

            ha_ndarray::ops::Op::enqueue(&op, &queue)
                .map(|buffer| buffer)
                .map_err(TCError::from)
        })
        .await
    }

    async fn construct_with_op<Ctr>(
        dir: DirLock<FE>,
        shape: Shape,
        block_ctr: Ctr,
    ) -> TCResult<Self>
    where
        Ctr: Fn(ha_ndarray::Context, ha_ndarray::Queue, usize) -> TCResult<Buffer<T>> + Copy,
    {
        shape.validate()?;

        let size = shape.size();
        let (block_size, num_blocks) = ideal_block_size_for(&shape);
        let block_axis = block_axis_for(&shape, block_size);
        let block_shape = block_shape_for(block_axis, &shape, block_size);

        let context = ha_ndarray::Context::default()?;
        let queue = ha_ndarray::Queue::new(context.clone(), block_size)?;

        let mut blocks = futures::stream::iter(0..num_blocks as u64)
            .map(|block_id| {
                let block_size = if (block_id + 1) * (block_size as u64) > size {
                    (size - (block_id * block_size as u64)) as usize
                } else {
                    block_size
                };

                let context = context.clone();
                let queue = queue.clone();
                async move {
                    block_ctr(context, queue, block_size).map(|buffer| (block_id, buffer))
                }
            })
            .buffered(num_cpus::get());

        let mut dir_guard = dir.write().await;

        let dtype_size = T::dtype().size();
        while let Some((block_id, buffer)) = blocks.try_next().await? {
            let size_in_bytes = dtype_size * buffer.len();
            dir_guard.create_file(block_id.to_string(), buffer, size_in_bytes)?;
        }

        std::mem::drop(dir_guard);

        let block_map = block_map_for(num_blocks as u64, &shape, &block_shape)?;

        Ok(Self {
            dir,
            block_map,
            block_size,
            shape,
            dtype: PhantomData,
        })
    }
}

impl<FE> DenseFile<FE, f32>
where
    FE: DenseCacheFile + AsType<Buffer<f32>>,
{
    pub async fn random_normal(
        dir: DirLock<FE>,
        shape: Shape,
        mean: f32,
        std: f32,
    ) -> TCResult<Self> {
        Self::construct_with_op(dir, shape, |context, queue, block_size| {
            let op = ha_ndarray::construct::RandomNormal::with_context(context, block_size)?;
            let random = ArrayOp::new(vec![block_size], op)
                .mul_scalar(std)?
                .add_scalar(mean)?;

            random
                .read(&queue)
                .and_then(|buffer| buffer.into_buffer())
                .map_err(TCError::from)
        })
        .await
    }

    pub async fn random_uniform(dir: DirLock<FE>, shape: Shape) -> TCResult<Self> {
        Self::construct_with_op(dir, shape, |context, queue, block_size| {
            let op = ha_ndarray::construct::RandomUniform::with_context(context, block_size)?;

            ha_ndarray::ops::Op::enqueue(&op, &queue)
                .map(|buffer| buffer)
                .map_err(TCError::from)
        })
        .await
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
    FE: AsType<Buffer<T>> + ThreadSafe,
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

        let blocks = futures::stream::iter(self.block_map.into_inner())
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
        let queue = ha_ndarray::Queue::new(block.context().clone(), block.size())?;
        let buffer = block.read(&queue)?;
        Ok(buffer.to_slice()?.as_ref()[block_offset])
    }
}

#[async_trait]
impl<'a, FE, T> DenseWrite for DenseFile<FE, T>
where
    FE: AsType<Buffer<T>> + ThreadSafe,
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

        let blocks = futures::stream::iter(self.block_map.into_inner())
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
    FE: AsType<Buffer<T>> + ThreadSafe,
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

macro_rules! impl_from_stream {
    ($t:ty, $decode:ident) => {
        #[async_trait]
        impl<FE> de::FromStream for DenseFile<FE, $t>
        where
            FE: AsType<Buffer<$t>> + ThreadSafe,
        {
            type Context = (DirLock<FE>, Shape);

            async fn from_stream<D: de::Decoder>(
                cxt: Self::Context,
                decoder: &mut D,
            ) -> Result<Self, D::Error> {
                let (dir, shape) = cxt;
                decoder.$decode(DenseVisitor::new(dir, shape)).await
            }
        }
    };
}

impl_from_stream!(f32, decode_array_f32);
impl_from_stream!(f64, decode_array_f64);
impl_from_stream!(i16, decode_array_i16);
impl_from_stream!(i32, decode_array_i32);
impl_from_stream!(i64, decode_array_i64);
impl_from_stream!(u8, decode_array_u8);
impl_from_stream!(u16, decode_array_u16);
impl_from_stream!(u32, decode_array_u32);
impl_from_stream!(u64, decode_array_u64);

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
        FE: AsType<Buffer<T>> + ThreadSafe,
        T: CDatatype + DType + 'static,
        Buffer<T>: de::FromStream<Context = ()>,
    {
        let num_blocks = div_ceil(self.shape.size(), self.block_size as u64);
        futures::stream::iter(0..num_blocks)
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
    FE: AsType<Buffer<T>> + ThreadSafe,
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
                    let mut block = dir.write_file(&block_id).await?;

                    let data = result?;
                    let data = data.read(&queue)?;
                    debug_assert_eq!(block.len(), data.len());
                    block.write(data)?;

                    Result::<(), TCError>::Ok(())
                }
            })
            .buffered(num_cpus::get())
            .try_fold((), |(), ()| futures::future::ready(Ok(())))
            .await
    }

    async fn overwrite_value(&self, _txn_id: TxnId, value: T) -> TCResult<()> {
        let num_blocks = div_ceil(self.shape.size(), self.block_size as u64);

        futures::stream::iter(0..num_blocks)
            .map(|block_id| async move {
                let mut block = self.dir.write_file(&block_id).await?;
                block.write_value(value).map_err(TCError::from)
            })
            .buffered(num_cpus::get())
            .try_fold((), |(), _| futures::future::ready(Ok(())))
            .await
    }

    async fn write_value(&self, _txn_id: TxnId, coord: Coord, value: T) -> TCResult<()> {
        self.shape.validate_coord(&coord)?;

        let offset = offset_of(coord, &self.shape);
        let block_id = offset / self.block_size as u64;

        let mut block = self.dir.write_file(&block_id).await?;
        block.write_value_at((offset % self.block_size as u64) as usize, value)?;

        Ok(())
    }
}

struct DenseVisitor<FE, T> {
    dir: DirLock<FE>,
    shape: Shape,
    dtype: PhantomData<T>,
}

impl<FE, T> DenseVisitor<FE, T> {
    fn new(dir: DirLock<FE>, shape: Shape) -> Self {
        Self {
            dir,
            shape,
            dtype: PhantomData,
        }
    }
}

macro_rules! impl_visitor {
    ($t:ty, $visit:ident) => {
        #[async_trait]
        impl<FE> de::Visitor for DenseVisitor<FE, $t>
        where
            FE: AsType<Buffer<$t>> + ThreadSafe,
        {
            type Value = DenseFile<FE, $t>;

            fn expecting() -> &'static str {
                "dense tensor data"
            }

            async fn $visit<A: de::ArrayAccess<$t>>(
                self,
                mut array: A,
            ) -> Result<Self::Value, A::Error> {
                let mut contents = self.dir.write().await;

                let (block_size, num_blocks) = ideal_block_size_for(&self.shape);

                let type_size = <$t>::dtype().size();
                let mut buffer = Vec::<$t>::with_capacity(block_size);
                for block_id in 0..num_blocks {
                    let size = array.buffer(&mut buffer).await?;
                    let block = Buffer::from(buffer.drain(..).collect::<Vec<$t>>());

                    contents
                        .create_file(block_id.to_string(), block, size * type_size)
                        .map_err(de::Error::custom)?;
                }

                std::mem::drop(contents);

                let block_axis = block_axis_for(&self.shape, block_size);
                let block_shape = block_shape_for(block_axis, &self.shape, block_size);
                let block_map = block_map_for(num_blocks as u64, &self.shape, &block_shape)
                    .map_err(de::Error::custom)?;

                Ok(DenseFile {
                    dir: self.dir,
                    block_map,
                    block_size,
                    shape: self.shape,
                    dtype: PhantomData,
                })
            }
        }
    };
}

impl_visitor!(f32, visit_array_f32);
impl_visitor!(f64, visit_array_f64);
impl_visitor!(i16, visit_array_i16);
impl_visitor!(i32, visit_array_i32);
impl_visitor!(i64, visit_array_i64);
impl_visitor!(u8, visit_array_u8);
impl_visitor!(u16, visit_array_u16);
impl_visitor!(u32, visit_array_u32);
impl_visitor!(u64, visit_array_u64);
