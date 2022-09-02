use std::fmt;
use std::iter::{self, FromIterator};
use std::marker::PhantomData;
use std::ops::{Add, Deref, Mul};

use afarray::{Array, ArrayExt, ArrayInstance, CoordBlocks, Coords, Offsets};
use arrayfire as af;
use async_trait::async_trait;
use destream::de;
use futures::stream::{self, Stream, StreamExt, TryStreamExt};
use futures::{future, try_join, TryFutureExt};
use log::debug;
use safecast::AsType;
use strided::Stride;

use tc_btree::{BTreeType, Node};
use tc_error::*;
use tc_transact::fs::{
    BlockId, CopyFrom, Dir, DirRead, DirWrite, File, FileRead, FileWrite, Persist, Restore,
};
use tc_transact::{Transact, Transaction, TxnId};
use tc_value::{Float, Number, NumberClass, NumberInstance, NumberType};
use tcgeneric::{TCBoxTryFuture, TCBoxTryStream};

use crate::stream::{Read, ReadValueAt};
use crate::{
    coord_bounds, transform, Bounds, Coord, FloatType, Schema, Shape, TensorAccess, TensorType,
};

use super::access::BlockListTranspose;
use super::{array_err, div_ceil, DenseAccess, DenseAccessor, DenseWrite, MEBIBYTE, PER_BLOCK};

/// The size of a dense tensor block on disk, in bytes (1 mebibyte + 5 bytes overhead).
const BLOCK_SIZE: usize = MEBIBYTE + 5;

/// A wrapper around a `DenseTensor` [`File`]
#[derive(Clone)]
pub struct BlockListFile<FD, FS, D, T> {
    file: FD,
    schema: Schema,
    sparse: PhantomData<FS>,
    dir: PhantomData<D>,
    txn: PhantomData<T>,
}

impl<FD, FS, D, T> BlockListFile<FD, FS, D, T> {
    fn new(file: FD, schema: Schema) -> Self {
        Self {
            file,
            schema,
            sparse: PhantomData,
            dir: PhantomData,
            txn: PhantomData,
        }
    }
}

impl<FD, FS, D, T> BlockListFile<FD, FS, D, T>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
{
    /// Construct a new `BlockListFile` with the given [`Shape`], filled with the given `value`.
    pub async fn constant(file: FD, txn_id: TxnId, shape: Shape, value: Number) -> TCResult<Self> {
        debug!("BlockListFile::constant {} {}", shape, value);

        let value_clone = value.clone();
        let generator = |length| Array::constant(value_clone.clone(), length);
        Self::from_blocks_generator(file, txn_id, shape, value.class(), generator).await
    }

    /// Construct a new `BlockListFile` with the given [`Shape`], with a random normal distribution.
    pub async fn random_normal(
        file: FD,
        txn_id: TxnId,
        shape: Shape,
        dtype: FloatType,
        mean: Float,
        std: Float,
    ) -> TCResult<Self> {
        debug!("BlockListFile::random_normal {}", shape);
        let generator = |length| {
            // TODO: impl Add<Number> and Mul<Number> for Array
            let std = Array::constant(std.into(), length);
            let mean = Array::constant(mean.into(), length);
            Array::random_normal(dtype, length).mul(&std).add(&mean)
        };

        Self::from_blocks_generator(file, txn_id, shape, dtype.into(), generator).await
    }

    /// Construct a new `BlockListFile` with the given [`Shape`], with a random normal distribution.
    pub async fn random_uniform(
        file: FD,
        txn_id: TxnId,
        shape: Shape,
        dtype: FloatType,
    ) -> TCResult<Self> {
        debug!("BlockListFile::random_normal {}", shape);
        let generator = |length| Array::random_uniform(dtype, length);
        Self::from_blocks_generator(file, txn_id, shape, dtype.into(), generator).await
    }

    async fn from_blocks_generator<G>(
        file: FD,
        txn_id: TxnId,
        shape: Shape,
        dtype: NumberType,
        generator: G,
    ) -> TCResult<Self>
    where
        G: Fn(usize) -> Array + Send + Copy,
    {
        let size = shape.size();

        let blocks = (0..(size / PER_BLOCK as u64)).map(move |_| Ok(generator(PER_BLOCK)));

        let trailing_len = (size % PER_BLOCK as u64) as usize;
        if trailing_len > 0 {
            let blocks = blocks.chain(iter::once(Ok(generator(trailing_len))));
            Self::from_blocks(file, txn_id, Some(shape), dtype, stream::iter(blocks)).await
        } else {
            Self::from_blocks(file, txn_id, Some(shape), dtype, stream::iter(blocks)).await
        }
    }

    /// Construct a new `BlockListFile` from the given `Stream` of [`Array`] blocks.
    pub async fn from_blocks<S: Stream<Item = TCResult<Array>> + Send + Unpin>(
        file: FD,
        txn_id: TxnId,
        shape: Option<Shape>,
        dtype: NumberType,
        mut blocks: S,
    ) -> TCResult<Self> {
        let mut file_lock = file.write(txn_id).await?;

        let bytes_per_element = dtype.size();
        let mut i = 0u64;
        let mut size = 0u64;
        // TODO: can this be parallelized?
        while let Some(block) = blocks.try_next().await? {
            let len = block.len();
            file_lock
                .create_block(i.into(), block, len * bytes_per_element)
                .await?;
            size += len as u64;
            i += 1;
        }

        let shape = if let Some(shape) = shape {
            if shape.size() < size {
                return Err(TCError::unsupported(format!(
                    "dense tensor with shape {} requires {} elements but found {}",
                    shape,
                    shape.size(),
                    size
                )));
            } else if shape.size() > size {
                return Err(TCError::unsupported(format!(
                    "dense tensor of shape {} requires {} elements but found {}--this could indicate a divide-by-zero error",
                    shape,
                    shape.size(),
                    size
                )));
            }

            shape
        } else {
            vec![size].into()
        };

        Ok(Self::new(file, Schema { shape, dtype }))
    }

    /// Construct a new `BlockListFile` from the given `Stream` of elements.
    pub async fn from_values<S: Stream<Item = TCResult<Number>> + Send + Unpin>(
        file: FD,
        txn_id: TxnId,
        shape: Shape,
        dtype: NumberType,
        values: S,
    ) -> TCResult<Self> {
        debug!("BlockListFile::from_values {}", shape);

        let mut i = 0u64;
        let mut size = 0u64;
        let mut values = values.chunks(PER_BLOCK);

        let mut file_lock = file.write(txn_id).await?;
        while let Some(chunk) = values.next().await {
            let chunk = chunk.into_iter().collect::<TCResult<Vec<Number>>>()?;
            size += chunk.len() as u64;

            let block_id = BlockId::from(i);
            let block = Array::from(chunk).cast_into(dtype);
            file_lock.create_block(block_id, block, BLOCK_SIZE).await?;

            i += 1;
        }

        if size != shape.size() {
            return Err(TCError::unsupported(format!(
                "DenseTensor of shape {} requires {} values, found {}",
                shape,
                shape.size(),
                size,
            )));
        }

        Ok(Self::new(file, Schema { shape, dtype }))
    }

    /// Construct a new `BlockListFile` of elements evenly distributed between `start` and `stop`.
    pub async fn range(
        file: FD,
        txn_id: TxnId,
        shape: Shape,
        start: Number,
        stop: Number,
    ) -> TCResult<Self> {
        let dtype = Ord::max(start.class(), stop.class());
        let step = (stop - start) / Number::from(shape.size() as f32);

        debug!(
            "{} tensor with range {} to {}, step {}",
            dtype, start, stop, step
        );

        let values = stream::iter(0..shape.size())
            .map(Number::from)
            .map(|i| start + (i * step))
            .map(Ok);

        Self::from_values(file, txn_id, shape, dtype, values).await
    }

    /// Sort the elements in this `BlockListFile`.
    pub async fn merge_sort(&self, txn_id: TxnId) -> TCResult<()> {
        let file = self.file.write(txn_id).await?;

        let num_blocks = div_ceil(self.size(), PER_BLOCK as u64);
        if num_blocks == 0 {
            return Ok(());
        } else if num_blocks == 1 {
            let block_id = BlockId::from(0u64);
            let mut block = file.write_block(block_id).await?;
            block.sort(true).map_err(array_err)?;
            return Ok(());
        }

        loop {
            let mut sorted = true;
            for block_id in 0..(num_blocks - 1) {
                let next_block_id = BlockId::from(block_id + 1);
                let block_id = BlockId::from(block_id);

                let left = file.write_block(block_id);
                let right = file.write_block(&next_block_id);
                let (mut left, mut right) = try_join!(left, right)?;

                let mut block = Array::concatenate(&left, &right);
                block.sort(true).map_err(array_err)?;

                let (left_sorted, right_sorted) = block.split(PER_BLOCK).map_err(array_err)?;
                sorted = sorted && left.deref() == &left_sorted && right.deref() == &right_sorted;

                *left = left_sorted;
                *right = right_sorted;
            }

            if sorted {
                return Ok(());
            }
        }
    }

    async fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCResult<()> {
        self.shape().validate_coord(&coord)?;

        let value = value.into_type(self.dtype());

        let offset: u64 = coord_bounds(self.shape())
            .iter()
            .zip(coord.iter())
            .map(|(d, x)| d * x)
            .sum();

        let block_id = BlockId::from(offset / PER_BLOCK as u64);
        let mut block = self.file.write_block(txn_id, block_id).await?;

        let offset = offset % PER_BLOCK as u64;

        (*block)
            .set_value(offset as usize, value)
            .map_err(array_err)
    }

    async fn overwrite<B: DenseAccess<FD, FS, D, T>>(&self, txn: T, value: B) -> TCResult<()> {
        if value.shape() != self.shape() {
            return Err(TCError::unsupported(format!(
                "cannot overwrite a Tensor of shape {} with one of shape {}",
                self.shape(),
                value.shape()
            )));
        }

        let txn_id = *txn.id();
        let (file, mut contents) = try_join!(self.file.read(txn_id), value.block_stream(txn))?;

        let mut block_id = 0u64;
        while let Some(array) = contents.try_next().await? {
            // TODO: can this be parallelized?
            let mut block = file.write_block(BlockId::from(block_id)).await?;
            *block = array;
            block_id += 1;
        }

        Ok(())
    }
}

impl<FD: Send, FS: Send, D: Send, T: Send> TensorAccess for BlockListFile<FD, FS, D, T> {
    fn dtype(&self) -> NumberType {
        self.schema.dtype
    }

    fn ndim(&self) -> usize {
        self.schema.shape.len()
    }

    fn shape(&'_ self) -> &'_ Shape {
        &self.schema.shape
    }

    fn size(&self) -> u64 {
        self.schema.shape.size()
    }
}

#[async_trait]
impl<FD, FS, D, T> DenseAccess<FD, FS, D, T> for BlockListFile<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    <D::Read as DirRead>::FileEntry: AsType<FD> + AsType<FS>,
    <D::Write as DirWrite>::FileClass: From<BTreeType> + From<TensorType>,
{
    type Slice = BlockListFileSlice<FD, FS, D, T>;
    type Transpose = BlockListTranspose<FD, FS, D, T, Self>;

    fn accessor(self) -> DenseAccessor<FD, FS, D, T> {
        DenseAccessor::File(self)
    }

    fn block_stream<'a>(self, txn: T) -> TCBoxTryFuture<'a, TCBoxTryStream<'a, Array>> {
        Box::pin(async move {
            let size = self.size();
            if size == 0 {
                let blocks: TCBoxTryStream<'a, Array> = Box::pin(stream::empty());
                return Ok(blocks);
            }

            let txn_id = *txn.id();

            let block_stream = Box::pin(
                stream::iter(0..(div_ceil(size, PER_BLOCK as u64)))
                    .map(BlockId::from)
                    .then(move |block_id| self.file.clone().read_block_owned(txn_id, block_id))
                    .map_ok(|block| (*block).clone()),
            );

            let block_stream: TCBoxTryStream<Array> = Box::pin(block_stream);
            Ok(block_stream)
        })
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        BlockListFileSlice::new(self, bounds)
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        BlockListTranspose::new(self, permutation)
    }

    async fn read_values(self, txn: Self::Txn, coords: Coords) -> TCResult<Array> {
        let txn_id = *txn.id();
        let per_block = ArrayExt::from(&[PER_BLOCK as u64][..]);

        let offsets = coords.to_offsets(self.shape());
        let block_offsets = &offsets / &per_block;
        let block_ids = block_offsets.unique(false).to_vec();

        let values = Array::constant(self.dtype().zero(), coords.len());

        stream::iter(block_ids)
            .map(|block_id| {
                let af_block_id = af::Array::new(&[block_id], af::Dim4::new(&[1, 1, 1, 1]));
                let mask = ArrayExt::from(af::eq(block_offsets.deref(), &af_block_id, true));
                let indices = (&offsets % &per_block).deref() * mask.deref();
                (block_id, mask.into(), indices.into())
            })
            .map(move |(block_id, mask, indices)| {
                self.file
                    .clone()
                    .read_block_owned(txn_id, BlockId::from(block_id))
                    .map_ok(move |block| block.get(&indices))
                    .map_ok(move |block_values| &block_values * &mask)
            })
            .buffer_unordered(num_cpus::get())
            .try_fold(values, |values, block_values| {
                future::ready(Ok(&values + &block_values))
            })
            .await
    }
}

#[async_trait]
impl<FD, FS, D, T> DenseWrite<FD, FS, D, T> for BlockListFile<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    <D::Read as DirRead>::FileEntry: AsType<FD> + AsType<FS>,
    <D::Write as DirWrite>::FileClass: From<BTreeType> + From<TensorType>,
{
    async fn write<B: DenseAccess<FD, FS, D, T>>(
        &self,
        txn: Self::Txn,
        bounds: Bounds,
        value: B,
    ) -> TCResult<()> {
        self.shape().validate_bounds(&bounds)?;

        if bounds == Bounds::all(self.shape()) {
            return self.overwrite(txn, value).await;
        }

        let slice_shape = bounds.to_shape(self.shape())?;
        if &slice_shape != value.shape() {
            return Err(TCError::unsupported(format!(
                "cannot overwrite a Tensor of shape {} with one of shape {}",
                slice_shape,
                value.shape()
            )));
        }

        let txn_id = *txn.id();
        let rebase = transform::Slice::new(self.shape().clone(), bounds)?;
        let size = rebase.size();
        let offsets = (0..size)
            .step_by(PER_BLOCK)
            .map(|start| {
                let end = start + PER_BLOCK as u64;
                if end > size {
                    Offsets::range(start, size)
                } else {
                    Offsets::range(start, end)
                }
            })
            .map(|offsets| Coords::from_offsets(offsets, rebase.shape()))
            .map(|coords| rebase.invert_coords(&coords).to_offsets(self.shape()));

        let blocks = value.block_stream(txn).await?;

        let af_per_block = af::constant(PER_BLOCK as u64, af::Dim4::new(&[1, 1, 1, 1]));
        stream::iter(offsets)
            .zip(blocks)
            .map(|(offsets, r)| r.map(|array| (offsets, array)))
            .map_ok(move |(offsets, array)| {
                let af_per_block = af_per_block.clone();

                async move {
                    let indices: ArrayExt<u64> =
                        af::modulo(offsets.deref(), &af_per_block, true).into();

                    let block_offsets = af::div(offsets.deref(), &af_per_block, true);
                    let block_ids = ArrayExt::from(af::set_unique(&block_offsets, true)).to_vec();

                    let mut start = 0;
                    for block_id in block_ids.into_iter() {
                        let af_block_id = ArrayExt::from(&[block_id][..]);
                        let (len, _) =
                            af::sum_all(&af::eq(&block_offsets, af_block_id.deref(), true));

                        let end = start + len as usize;
                        let indices = indices.slice(start, end);
                        let array = array.slice(start, end).map_err(array_err)?;

                        let mut block = self
                            .file
                            .write_block(txn_id, BlockId::from(block_id))
                            .await?;

                        block.set(&indices, &array).map_err(array_err)?;

                        start = end;
                    }

                    Ok(())
                }
            })
            .try_buffer_unordered(num_cpus::get())
            .try_fold((), |(), ()| future::ready(Ok(())))
            .await
    }

    async fn write_value(&self, txn_id: TxnId, mut bounds: Bounds, value: Number) -> TCResult<()> {
        debug!("BlockListFile::write_value {} at {}", value, bounds);
        self.shape().validate_bounds(&bounds)?;

        if let Some(coord) = bounds.as_coord(self.shape()) {
            return self.write_value_at(txn_id, coord, value).await;
        }

        bounds.normalize(self.shape());

        let file = &self.file.read(txn_id).await?;
        let coords = stream::iter(bounds.affected().map(TCResult::Ok));
        CoordBlocks::new(coords, bounds.len(), PER_BLOCK)
            .map_ok(|coords| {
                let (block_ids, af_indices, af_offsets) = coord_block(coords, self.shape());

                let value = value.clone();

                async move {
                    let mut start = 0;
                    for block_id in block_ids {
                        let value = value.clone();
                        let (block_offsets, new_start) =
                            block_offsets(&af_indices, &af_offsets, start, block_id);

                        let block_id = BlockId::from(block_id);
                        let mut block = file.write_block(block_id).await?;

                        let value = Array::constant(value, (new_start - start) as usize);
                        (*block)
                            .set(&block_offsets.into(), &value)
                            .map_err(array_err)?;

                        start = new_start;
                    }

                    Ok(())
                }
            })
            .try_buffer_unordered(num_cpus::get())
            .try_fold((), |(), ()| future::ready(Ok(())))
            .await
    }
}

impl<FD, FS, D, T> ReadValueAt<D> for BlockListFile<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    <D::Read as DirRead>::FileEntry: AsType<FD> + AsType<FS>,
    <D::Write as DirWrite>::FileClass: From<BTreeType> + From<TensorType>,
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: T, coord: Coord) -> Read<'a> {
        Box::pin(async move {
            self.shape().validate_coord(&coord)?;

            let offset: u64 = coord_bounds(self.shape())
                .iter()
                .zip(coord.iter())
                .map(|(d, x)| d * x)
                .sum();

            let block_id = BlockId::from(offset / PER_BLOCK as u64);
            let block = self.file.read_block(*txn.id(), block_id).await?;
            let value = block.get_value((offset % PER_BLOCK as u64) as usize);

            Ok((coord, value))
        })
    }
}

#[async_trait]
impl<FD, FS, D, T> Transact for BlockListFile<FD, FS, D, T>
where
    FD: File<Array> + Transact,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
{
    async fn commit(&self, txn_id: &TxnId) {
        self.file.commit(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.file.finalize(txn_id).await
    }
}

#[async_trait]
impl<FD, FS, D, T, B> CopyFrom<D, B> for BlockListFile<FD, FS, D, T>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
    B: DenseAccess<FD, FS, D, T>,
{
    async fn copy_from(other: B, file: FD, txn: &T) -> TCResult<Self> {
        let txn_id = *txn.id();
        let dtype = other.dtype();
        let shape = other.shape().clone();
        let blocks = other.block_stream(txn.clone()).await?;
        Self::from_blocks(file, txn_id, Some(shape), dtype, blocks).await
    }
}

#[async_trait]
impl<FD, FS, D, T> Persist<D> for BlockListFile<FD, FS, D, T>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
{
    type Schema = Schema;
    type Store = FD;
    type Txn = T;

    fn schema(&self) -> &Schema {
        &self.schema
    }

    async fn load(_txn: &T, schema: Self::Schema, file: Self::Store) -> TCResult<Self> {
        Ok(Self::new(file, schema))
    }
}

#[async_trait]
impl<FD, FS, D, T> Restore<D> for BlockListFile<FD, FS, D, T>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
{
    async fn restore(&self, backup: &Self, txn_id: TxnId) -> TCResult<()> {
        if self.schema.shape != backup.schema.shape {
            return Err(TCError::bad_request(
                "cannot restore a dense Tensor from a backup with a different shape",
                &backup.schema.shape,
            ));
        }

        if self.schema.dtype != backup.schema.dtype {
            return Err(TCError::bad_request(
                "cannot restore a dense Tensor from a backup with a different data type",
                &backup.schema.dtype,
            ));
        }

        self.file.copy_from(txn_id, &backup.file, false).await
    }
}

impl<FD, FS, D, T> fmt::Display for BlockListFile<FD, FS, D, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("dense Tensor file")
    }
}

#[async_trait]
impl<FD, FS, D, T> de::FromStream for BlockListFile<FD, FS, D, T>
where
    FD: File<Array>,
    FS: Send,
    D: Send,
    T: Send,
{
    type Context = (TxnId, FD, Schema);

    async fn from_stream<De: de::Decoder>(
        cxt: (TxnId, FD, Schema),
        decoder: &mut De,
    ) -> Result<Self, De::Error> {
        let (txn_id, file, schema) = cxt;
        let visitor = BlockListVisitor::new(txn_id, &file);

        use tc_value::{
            ComplexType as CT, FloatType as FT, IntType as IT, NumberType as NT, UIntType as UT,
        };

        fn err_nonspecific<T: fmt::Display, E: de::Error>(class: T) -> E {
            de::Error::custom(format!(
                "tensor does not support {} (use a more specific type)",
                class
            ))
        }

        let size = match schema.dtype {
            NT::Bool => decoder.decode_array_bool(visitor).await,
            NT::Complex(ct) => match ct {
                CT::C32 => {
                    decoder
                        .decode_array_f32(ComplexBlockListVisitor { visitor })
                        .await
                }
                CT::C64 => {
                    decoder
                        .decode_array_f64(ComplexBlockListVisitor { visitor })
                        .await
                }
                CT::Complex => Err(err_nonspecific(CT::Complex)),
            },
            NT::Float(ft) => match ft {
                FT::F32 => decoder.decode_array_f32(visitor).await,
                FT::F64 => decoder.decode_array_f64(visitor).await,
                FT::Float => Err(err_nonspecific(FT::Float)),
            },
            NT::Int(it) => match it {
                IT::I8 => Err(de::Error::custom("tensor does not support 8-bit integer")),
                IT::I16 => decoder.decode_array_i16(visitor).await,
                IT::I32 => decoder.decode_array_i32(visitor).await,
                IT::I64 => decoder.decode_array_i64(visitor).await,
                IT::Int => Err(err_nonspecific(FT::Float)),
            },
            NT::UInt(ut) => match ut {
                UT::U8 => decoder.decode_array_u8(visitor).await,
                UT::U16 => decoder.decode_array_u16(visitor).await,
                UT::U32 => decoder.decode_array_u32(visitor).await,
                UT::U64 => decoder.decode_array_u64(visitor).await,
                UT::UInt => Err(err_nonspecific(FT::Float)),
            },
            NT::Number => Err(err_nonspecific(NT::Number)),
        }?;

        if size == schema.shape.size() {
            Ok(Self::new(file, schema))
        } else {
            Err(de::Error::custom(format!(
                "tensor data has the wrong number of elements ({}) for shape {}",
                size, &schema.shape
            )))
        }
    }
}

struct BlockListVisitor<'a, F> {
    txn_id: TxnId,
    file: &'a F,
}

impl<'a, F: File<Array>> BlockListVisitor<'a, F> {
    fn new(txn_id: TxnId, file: &'a F) -> Self {
        Self { txn_id, file }
    }
}

impl<'a, F: File<Array>> BlockListVisitor<'a, F> {
    async fn visit_array<
        T: af::HasAfEnum + Clone + Copy + Default,
        A: de::ArrayAccess<T>,
        const BUF_SIZE: usize,
    >(
        &self,
        mut access: A,
    ) -> Result<u64, A::Error>
    where
        Array: From<ArrayExt<T>>,
    {
        debug!(
            "BlockListVisitor decoding array of type {}",
            std::any::type_name::<T>()
        );

        let mut buf = vec![T::default(); BUF_SIZE];
        let mut size = 0u64;
        let mut block_id = 0u64;

        let mut file = self
            .file
            .write(self.txn_id)
            .map_err(de::Error::custom)
            .await?;

        loop {
            let block_size = access.buffer(&mut buf).await?;

            if block_size == 0 {
                break;
            } else {
                let block = ArrayExt::from(&buf[..block_size]);
                file.create_block(block_id.into(), block.into(), BLOCK_SIZE)
                    .map_err(de::Error::custom)
                    .await?;
                size += block_size as u64;
                block_id += 1;
            }
        }

        Ok(size)
    }
}

#[async_trait]
impl<'a, F: File<Array>> de::Visitor for BlockListVisitor<'a, F> {
    type Value = u64;

    fn expecting() -> &'static str {
        "tensor data"
    }

    async fn visit_array_bool<A: de::ArrayAccess<bool>>(
        self,
        access: A,
    ) -> Result<Self::Value, A::Error> {
        self.visit_array::<bool, A, MEBIBYTE>(access).await
    }

    async fn visit_array_i16<A: de::ArrayAccess<i16>>(
        self,
        access: A,
    ) -> Result<Self::Value, A::Error> {
        const BUF_SIZE: usize = MEBIBYTE / 2;
        self.visit_array::<i16, A, BUF_SIZE>(access).await
    }

    async fn visit_array_i32<A: de::ArrayAccess<i32>>(
        self,
        access: A,
    ) -> Result<Self::Value, A::Error> {
        const BUF_SIZE: usize = MEBIBYTE / 4;
        self.visit_array::<i32, A, BUF_SIZE>(access).await
    }

    async fn visit_array_i64<A: de::ArrayAccess<i64>>(
        self,
        access: A,
    ) -> Result<Self::Value, A::Error> {
        const BUF_SIZE: usize = MEBIBYTE / 8;
        self.visit_array::<i64, A, BUF_SIZE>(access).await
    }

    async fn visit_array_u8<A: de::ArrayAccess<u8>>(
        self,
        access: A,
    ) -> Result<Self::Value, A::Error> {
        self.visit_array::<u8, A, MEBIBYTE>(access).await
    }

    async fn visit_array_u16<A: de::ArrayAccess<u16>>(
        self,
        access: A,
    ) -> Result<Self::Value, A::Error> {
        const BUF_SIZE: usize = MEBIBYTE / 2;
        self.visit_array::<u16, A, BUF_SIZE>(access).await
    }

    async fn visit_array_u32<A: de::ArrayAccess<u32>>(
        self,
        access: A,
    ) -> Result<Self::Value, A::Error> {
        const BUF_SIZE: usize = MEBIBYTE / 4;
        self.visit_array::<u32, A, BUF_SIZE>(access).await
    }

    async fn visit_array_u64<A: de::ArrayAccess<u64>>(
        self,
        access: A,
    ) -> Result<Self::Value, A::Error> {
        const BUF_SIZE: usize = MEBIBYTE / 8;
        self.visit_array::<u64, A, BUF_SIZE>(access).await
    }

    async fn visit_array_f32<A: de::ArrayAccess<f32>>(
        self,
        access: A,
    ) -> Result<Self::Value, A::Error> {
        const BUF_SIZE: usize = MEBIBYTE / 4;
        self.visit_array::<f32, A, BUF_SIZE>(access).await
    }

    async fn visit_array_f64<A: de::ArrayAccess<f64>>(
        self,
        access: A,
    ) -> Result<Self::Value, A::Error> {
        const BUF_SIZE: usize = MEBIBYTE / 8;
        self.visit_array::<f64, A, BUF_SIZE>(access).await
    }
}

struct ComplexBlockListVisitor<'a, F> {
    visitor: BlockListVisitor<'a, F>,
}

impl<'a, F: File<Array>> ComplexBlockListVisitor<'a, F> {
    async fn visit_array<
        C: af::HasAfEnum,
        T: af::HasAfEnum + Clone + Copy + Default,
        A: de::ArrayAccess<T>,
        const BUF_SIZE: usize,
    >(
        &self,
        mut access: A,
    ) -> Result<u64, A::Error>
    where
        ArrayExt<C>: From<(ArrayExt<T>, ArrayExt<T>)>,
        Array: From<ArrayExt<C>>,
    {
        let mut buf = [T::default(); BUF_SIZE];
        let mut size = 0u64;
        let mut block_id = 0u64;

        let mut file = self
            .visitor
            .file
            .write(self.visitor.txn_id)
            .map_err(de::Error::custom)
            .await?;

        loop {
            let block_size = access.buffer(&mut buf).await?;

            if block_size == 0 {
                break;
            } else {
                let (re, im) = Stride::new(&buf).substrides2();
                let re = ArrayExt::<T>::from_iter(re.iter().cloned());
                let im = ArrayExt::<T>::from_iter(im.iter().cloned());
                let block = ArrayExt::from((re, im));
                file.create_block(block_id.into(), block.into(), BLOCK_SIZE)
                    .map_err(de::Error::custom)
                    .await?;

                size += block_size as u64;
                block_id += 1;
            }
        }

        Ok(size)
    }
}

#[async_trait]
impl<'a, F: File<Array>> de::Visitor for ComplexBlockListVisitor<'a, F> {
    type Value = u64;

    fn expecting() -> &'static str {
        "complex tensor data"
    }

    async fn visit_array_f32<A: de::ArrayAccess<f32>>(
        self,
        access: A,
    ) -> Result<Self::Value, A::Error> {
        const BUF_SIZE: usize = (MEBIBYTE / 4) * 2;
        self.visit_array::<_, f32, A, BUF_SIZE>(access).await
    }

    async fn visit_array_f64<A: de::ArrayAccess<f64>>(
        self,
        access: A,
    ) -> Result<Self::Value, A::Error> {
        const BUF_SIZE: usize = (MEBIBYTE / 8) * 2;
        self.visit_array::<_, f64, A, BUF_SIZE>(access).await
    }
}

/// A slice of a [`BlockListFile`]
#[derive(Clone)]
pub struct BlockListFileSlice<FD, FS, D, T> {
    source: BlockListFile<FD, FS, D, T>,
    rebase: transform::Slice,
}

impl<FD, FS, D, T> BlockListFileSlice<FD, FS, D, T>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
{
    fn new(source: BlockListFile<FD, FS, D, T>, bounds: Bounds) -> TCResult<Self> {
        debug!(
            "BlockListFile shape {} slice bounds {}",
            source.shape(),
            bounds
        );

        let rebase = transform::Slice::new(source.shape().clone(), bounds)?;
        debug!("BlockListFileSlice shape is {}", rebase.shape());
        Ok(Self { source, rebase })
    }
}

impl<FD, FS, D, T> TensorAccess for BlockListFileSlice<FD, FS, D, T>
where
    FD: File<Array>,
    FS: File<Node>,
    D: Dir,
    T: Transaction<D>,
{
    fn dtype(&self) -> NumberType {
        self.source.dtype()
    }

    fn ndim(&self) -> usize {
        self.rebase.ndim()
    }

    fn shape(&self) -> &Shape {
        self.rebase.shape()
    }

    fn size(&self) -> u64 {
        self.rebase.size()
    }
}

#[async_trait]
impl<FD, FS, D, T> DenseAccess<FD, FS, D, T> for BlockListFileSlice<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    <D::Read as DirRead>::FileEntry: AsType<FD> + AsType<FS>,
    <D::Write as DirWrite>::FileClass: From<BTreeType> + From<TensorType>,
{
    type Slice = Self;
    type Transpose = BlockListTranspose<FD, FS, D, T, Self>;

    fn accessor(self) -> DenseAccessor<FD, FS, D, T> {
        DenseAccessor::Slice(self)
    }

    fn block_stream<'a>(self, txn: T) -> TCBoxTryFuture<'a, TCBoxTryStream<'a, Array>> {
        if self.size() == 0 {
            let blocks: TCBoxTryStream<'a, Array> = Box::pin(stream::empty());
            return Box::pin(future::ready(Ok(blocks)));
        }

        Box::pin(async move {
            let txn_id = *txn.id();
            let file = self.source.file.clone();
            let shape = self.source.schema.shape;
            let mut bounds = self.rebase.bounds().clone();
            bounds.normalize(&shape);

            let ndim = bounds.len();
            let coords = stream::iter(bounds.affected().map(TCResult::Ok));
            let blocks = CoordBlocks::new(coords, ndim, PER_BLOCK).and_then(move |coords| {
                let (block_ids, af_indices, af_offsets) = coord_block(coords, &shape);
                let file_clone = file.clone();

                Box::pin(async move {
                    let mut start = 0;
                    let mut values = vec![];
                    for block_id in block_ids {
                        let (block_offsets, new_start) =
                            block_offsets(&af_indices, &af_offsets, start, block_id);

                        let block = file_clone
                            .read_block(txn_id, BlockId::from(block_id))
                            .await?;

                        values.extend(block.get(&block_offsets.into()).to_vec());
                        start = new_start;
                    }

                    Ok(Array::from(values))
                })
            });

            let blocks: TCBoxTryStream<Array> = Box::pin(blocks);
            Ok(blocks)
        })
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        let bounds = self.rebase.invert_bounds(bounds);
        let slice = self.source.slice(bounds)?;
        Ok(slice)
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        BlockListTranspose::new(self, permutation)
    }

    async fn read_values(self, txn: Self::Txn, coords: Coords) -> TCResult<Array> {
        let coords = self.rebase.invert_coords(&coords);
        self.source.read_values(txn, coords).await
    }
}

impl<FD, FS, D, T> ReadValueAt<D> for BlockListFileSlice<FD, FS, D, T>
where
    D: Dir,
    T: Transaction<D>,
    FD: File<Array>,
    FS: File<Node>,
    <D::Read as DirRead>::FileEntry: AsType<FD> + AsType<FS>,
    <D::Write as DirWrite>::FileClass: From<BTreeType> + From<TensorType>,
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: Self::Txn, coord: Coord) -> Read<'a> {
        Box::pin(async move {
            self.shape().validate_coord(&coord)?;
            let source_coord = self.rebase.invert_coord(&coord);
            let (_, value) = self.source.read_value_at(txn, source_coord).await?;
            Ok((coord, value))
        })
    }
}

impl<FD, FS, D, T> fmt::Display for BlockListFileSlice<FD, FS, D, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("dense Tensor slice")
    }
}

fn block_offsets(
    indices: &ArrayExt<u64>,
    offsets: &ArrayExt<u64>,
    start: usize,
    block_id: u64,
) -> (ArrayExt<u64>, usize) {
    assert_eq!(indices.len(), offsets.len());

    let num_to_update = af::sum_all(&af::eq(
        indices.deref(),
        &af::constant(block_id, af::Dim4::new(&[1, 1, 1, 1])),
        true,
    ))
    .0;

    if num_to_update == 0 {
        return (af::Array::new_empty(af::Dim4::default()).into(), start);
    }

    let end = start + num_to_update as usize;
    let block_offsets = offsets.slice(start, end);

    (block_offsets, end)
}

fn coord_block(coords: Coords, shape: &[u64]) -> (Vec<u64>, ArrayExt<u64>, Offsets) {
    let af_per_block = af::constant(PER_BLOCK as u64, af::Dim4::new(&[1, 1, 1, 1]));

    let offsets = coords.to_offsets(shape);
    let block_offsets = ArrayExt::from(af::div(offsets.deref(), &af_per_block, true));
    let block_ids = block_offsets.unique(true);
    (block_ids.to_vec(), block_offsets, offsets)
}
