use std::convert::TryFrom;
use std::fmt;
use std::iter::{self, FromIterator};
use std::marker::PhantomData;

use afarray::{Array, ArrayExt};
use arrayfire as af;
use async_trait::async_trait;
use destream::de;
use futures::stream::{self, Stream, StreamExt, TryStreamExt};
use futures::{future, try_join, TryFutureExt};
use log::debug;
use number_general::{Number, NumberClass, NumberInstance, NumberType};
use strided::Stride;

use tc_btree::Node;
use tc_error::*;
use tc_transact::fs::{BlockData, BlockId, CopyFrom, Dir, File, Persist, Restore};
use tc_transact::{Transact, Transaction, TxnId};
use tc_value::Value;
use tcgeneric::{TCBoxTryFuture, TCTryStream};

use crate::stream::{block_offsets, coord_block, coord_bounds, Read, ReadValueAt};
use crate::transform::{self, Rebase};
use crate::{Bounds, Coord, Schema, Shape, TensorAccess, TensorType};

use super::access::BlockListTranspose;
use super::{DenseAccess, DenseAccessor, PER_BLOCK};

const MEBIBYTE: usize = 1_048_576;

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
        if !file.is_empty(&txn_id).await? {
            return Err(TCError::unsupported(
                "cannot create new tensor: file is not empty",
            ));
        }

        let size = shape.size();

        let value_clone = value.clone();
        let blocks = (0..(size / PER_BLOCK as u64))
            .map(move |_| Ok(Array::constant(value_clone.clone(), PER_BLOCK)));

        let trailing_len = (size % PER_BLOCK as u64) as usize;
        if trailing_len > 0 {
            let blocks = blocks.chain(iter::once(Ok(Array::constant(value.clone(), trailing_len))));
            Self::from_blocks(file, txn_id, shape, value.class(), stream::iter(blocks)).await
        } else {
            Self::from_blocks(file, txn_id, shape, value.class(), stream::iter(blocks)).await
        }
    }

    /// Construct a new `BlockListFile` from the given `Stream` of [`Array`] blocks.
    pub async fn from_blocks<S: Stream<Item = TCResult<Array>> + Send + Unpin>(
        file: FD,
        txn_id: TxnId,
        shape: Shape,
        dtype: NumberType,
        blocks: S,
    ) -> TCResult<Self> {
        debug!("BlockListFile::from_blocks {}", shape);

        blocks
            .enumerate()
            .map(|(i, r)| r.map(|block| (BlockId::from(i), block)))
            .inspect_ok(|(id, block)| debug!("block {} has {} elements", id, block.len()))
            .map_ok(|(id, block)| file.create_block(txn_id, id, block))
            .try_buffer_unordered(num_cpus::get())
            .try_fold((), |_, _| future::ready(Ok(())))
            .await?;

        if file.is_empty(&txn_id).await? {
            return Err(TCError::unsupported(
                "tried to create a dense tensor from an empty block list",
            ));
        }

        Ok(Self::new(file, (shape, dtype)))
    }

    /// Construct a new `BlockListFile` from the given `Stream` of elements.
    pub async fn from_values<S: Stream<Item = Number> + Send + Unpin>(
        file: FD,
        txn_id: TxnId,
        shape: Shape,
        dtype: NumberType,
        values: S,
    ) -> TCResult<Self> {
        let mut i = 0u64;
        let mut values = values.chunks((Array::max_size() as usize) / dtype.size());
        while let Some(chunk) = values.next().await {
            let block_id = BlockId::from(i);
            let block = Array::from(chunk).cast_into(dtype);
            file.create_block(txn_id, block_id, block).await?;
            i += 1;
        }

        Ok(Self::new(file, (shape, dtype)))
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
            .map(|i| start + (i * step));

        Self::from_values(file, txn_id, shape, dtype, values).await
    }

    /// Consume this `BlockListFile` handle and return a `Stream` of `Array` blocks.
    pub fn into_stream(self, txn_id: TxnId) -> impl Stream<Item = TCResult<Array>> + Unpin {
        let num_blocks = div_ceil(self.size(), PER_BLOCK as u64);
        let blocks = stream::iter((0..num_blocks).into_iter().map(BlockId::from))
            .then(move |block_id| self.file.clone().read_block_owned(txn_id, block_id))
            .map_ok(|block| (*block).clone());

        Box::pin(blocks)
    }

    /// Sort the elements in this `BlockListFile`.
    pub async fn merge_sort(&self, txn_id: TxnId) -> TCResult<()> {
        let num_blocks = div_ceil(self.size(), PER_BLOCK as u64);
        if num_blocks <= 1 {
            let block_id = BlockId::from(0u64);
            let mut block = self.file.write_block(txn_id, block_id).await?;
            block.sort(true)?;
            return Ok(());
        }

        for block_id in 0..(num_blocks - 1) {
            let next_block_id = BlockId::from(block_id + 1);
            let block_id = BlockId::from(block_id);

            let left = self.file.write_block(txn_id, block_id);
            let right = self.file.write_block(txn_id, next_block_id);
            let (mut left, mut right) = try_join!(left, right)?;

            let mut block = Array::concatenate(&left, &right)?;
            block.sort(true)?;

            let (left_sorted, right_sorted) = block.split(PER_BLOCK)?;
            *left = left_sorted;
            *right = right_sorted;
        }

        Ok(())
    }
}

impl<FD: Send, FS: Send, D: Send, T: Send> TensorAccess for BlockListFile<FD, FS, D, T> {
    fn dtype(&self) -> NumberType {
        self.schema.1
    }

    fn ndim(&self) -> usize {
        self.schema.0.len()
    }

    fn shape(&'_ self) -> &'_ Shape {
        &self.schema.0
    }

    fn size(&self) -> u64 {
        self.schema.0.size()
    }
}

#[async_trait]
impl<FD, FS, D, T> DenseAccess<FD, FS, D, T> for BlockListFile<FD, FS, D, T>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node> + TryFrom<D::File, Error = TCError>,
    D: Dir,
    T: Transaction<D>,
    D::FileClass: From<TensorType>,
{
    type Slice = BlockListFileSlice<FD, FS, D, T>;
    type Transpose = BlockListTranspose<FD, FS, D, T, Self>;

    fn accessor(self) -> DenseAccessor<FD, FS, D, T> {
        DenseAccessor::File(self)
    }

    fn block_stream<'a>(self, txn: T) -> TCBoxTryFuture<'a, TCTryStream<'a, Array>> {
        Box::pin(async move {
            let size = self.size();
            let file = self.file;
            let block_stream = Box::pin(
                stream::iter(0..(div_ceil(size, PER_BLOCK as u64)))
                    .map(BlockId::from)
                    .then(move |block_id| file.clone().read_block_owned(*txn.id(), block_id))
                    .map_ok(|block| (*block).clone()),
            );

            let block_stream: TCTryStream<Array> = Box::pin(block_stream);
            Ok(block_stream)
        })
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        BlockListFileSlice::new(self, bounds)
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        BlockListTranspose::new(self, permutation)
    }

    async fn write_value(&self, txn_id: TxnId, mut bounds: Bounds, value: Number) -> TCResult<()> {
        debug!("BlockListFile::write_value {} at {}", value, bounds);

        if !self.shape().contains_bounds(&bounds) {
            return Err(TCError::bad_request("Bounds out of bounds", bounds));
        } else if bounds.len() == self.ndim() {
            if let Some(coord) = bounds.as_coord() {
                return self.write_value_at(txn_id, coord, value).await;
            }
        }

        bounds.normalize(self.shape());
        let coord_bounds = coord_bounds(self.shape());

        stream::iter(bounds.affected())
            .chunks(PER_BLOCK)
            .map(|coords| {
                let ndim = coords[0].len();
                let num_coords = coords.len() as u64;
                let (block_ids, af_indices, af_offsets) = coord_block(
                    coords.into_iter(),
                    &coord_bounds,
                    PER_BLOCK,
                    ndim,
                    num_coords,
                );

                let file = &self.file;
                let value = value.clone();
                let txn_id = txn_id;

                Ok(async move {
                    let mut start = 0f64;
                    for block_id in block_ids {
                        let value = value.clone();
                        let (block_offsets, new_start) =
                            block_offsets(&af_indices, &af_offsets, start, block_id);

                        let block_id = BlockId::from(block_id);
                        let mut block = file.write_block(txn_id, block_id).await?;

                        let value = Array::constant(value, (new_start - start) as usize);
                        (*block).set(&block_offsets.into(), &value)?;
                        start = new_start;
                    }

                    Ok(())
                })
            })
            .try_buffer_unordered(num_cpus::get())
            .try_fold((), |(), ()| future::ready(Ok(())))
            .await
    }

    fn write_value_at(&self, txn_id: TxnId, coord: Coord, value: Number) -> TCBoxTryFuture<()> {
        Box::pin(async move {
            debug!("BlockListFile::write_value_at {:?} <- {}", coord, value);

            if !self.shape().contains_coord(&coord) {
                return Err(TCError::bad_request(
                    "Invalid coordinate",
                    format!("[{:?}]", coord),
                ));
            }

            let value = value.into_type(self.dtype());

            let offset: u64 = coord_bounds(self.shape())
                .iter()
                .zip(coord.iter())
                .map(|(d, x)| d * x)
                .sum();

            let block_id = BlockId::from(offset / PER_BLOCK as u64);
            let mut block = self.file.write_block(txn_id, block_id).await?;

            let offset = offset % PER_BLOCK as u64;
            debug!("offset is {}", offset);

            (*block)
                .set_value(offset as usize, value)
                .map_err(TCError::from)
        })
    }
}

impl<FD, FS, D, T> ReadValueAt<D> for BlockListFile<FD, FS, D, T>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node> + TryFrom<D::File, Error = TCError>,
    D: Dir,
    T: Transaction<D>,
    D::FileClass: From<TensorType>,
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: T, coord: Coord) -> Read<'a> {
        Box::pin(async move {
            debug!(
                "read value at {:?} from BlockListFile with shape {}",
                coord,
                self.shape()
            );

            if !self.shape().contains_coord(&coord) {
                return Err(TCError::bad_request(
                    "Coordinate is out of bounds",
                    Value::from_iter(coord),
                ));
            }

            let offset: u64 = coord_bounds(self.shape())
                .iter()
                .zip(coord.iter())
                .map(|(d, x)| d * x)
                .sum();

            debug!("coord {:?} is offset {}", coord, offset);

            let block_id = BlockId::from(offset / PER_BLOCK as u64);
            let block = self.file.read_block(*txn.id(), block_id).await?;

            debug!(
                "read offset {} from block of length {}",
                offset % PER_BLOCK as u64,
                block.len()
            );

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
    async fn copy_from(other: B, file: FD, txn: T) -> TCResult<Self> {
        let txn_id = *txn.id();
        let dtype = other.dtype();
        let shape = other.shape().clone();
        let blocks = other.block_stream(txn).await?;
        Self::from_blocks(file, txn_id, shape, dtype, blocks).await
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
        if self.schema.0 != backup.schema.0 {
            return Err(TCError::bad_request(
                "cannot restore a dense Tensor from a backup with a different shape",
                &backup.schema.0,
            ));
        }

        if self.schema.1 != backup.schema.1 {
            return Err(TCError::bad_request(
                "cannot restore a dense Tensor from a backup with a different data type",
                &backup.schema.1,
            ));
        }

        self.file.truncate(txn_id).await?;
        self.file.copy_from(&backup.file, txn_id).await
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
        let (txn_id, file, (shape, dtype)) = cxt;
        let visitor = BlockListVisitor::new(txn_id, &file);

        use number_general::{
            ComplexType as CT, FloatType as FT, IntType as IT, NumberType as NT, UIntType as UT,
        };

        fn err_nonspecific<T: fmt::Display, E: de::Error>(class: T) -> E {
            de::Error::custom(format!(
                "tensor does not support {} (use a more specific type)",
                class
            ))
        }

        let size = match dtype {
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

        if size == shape.size() {
            Ok(Self::new(file, (shape, dtype)))
        } else {
            Err(de::Error::custom(format!(
                "tensor data has the wrong number of elements ({}) for shape {}",
                size, shape
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

    async fn create_block<T: af::HasAfEnum, E: de::Error>(
        &self,
        block_id: u64,
        block: ArrayExt<T>,
    ) -> Result<<F as File<Array>>::Block, E>
    where
        Array: From<ArrayExt<T>>,
    {
        debug!("BlockListVisitor::create_block {}", block_id);

        self.file
            .create_block(self.txn_id, block_id.into(), block.into())
            .map_err(de::Error::custom)
            .await
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
        let mut buf = vec![T::default(); BUF_SIZE];
        let mut size = 0u64;
        let mut block_id = 0u64;

        loop {
            let block_size = access.buffer(&mut buf).await?;

            if block_size == 0 {
                break;
            } else {
                let block = ArrayExt::from(&buf[..block_size]);
                self.create_block(block_id, block).await?;
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

        loop {
            let block_size = access.buffer(&mut buf).await?;

            if block_size == 0 {
                break;
            } else {
                let (re, im) = Stride::new(&buf).substrides2();
                let re = ArrayExt::<T>::from_iter(re.iter().cloned());
                let im = ArrayExt::<T>::from_iter(im.iter().cloned());
                let block = ArrayExt::from((re, im));
                self.visitor.create_block(block_id, block).await?;
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
        let rebase = transform::Slice::new(source.shape().clone(), bounds)?;
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
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node> + TryFrom<D::File, Error = TCError>,
    D: Dir,
    T: Transaction<D>,
    D::FileClass: From<TensorType>,
{
    type Slice = Self;
    type Transpose = BlockListTranspose<FD, FS, D, T, Self>;

    fn accessor(self) -> DenseAccessor<FD, FS, D, T> {
        DenseAccessor::Slice(self)
    }

    fn block_stream<'a>(self, txn: T) -> TCBoxTryFuture<'a, TCTryStream<'a, Array>> {
        let txn_id = *txn.id();
        let file = self.source.file;
        let shape = self.source.schema.0;
        let mut bounds = self.rebase.bounds().clone();
        bounds.normalize(&shape);
        let coord_bounds = coord_bounds(&shape);

        let values = stream::iter(bounds.affected())
            .chunks(PER_BLOCK)
            .then(move |coords| {
                let ndim = coords[0].len();
                let num_coords = coords.len() as u64;
                let file_clone = file.clone();
                let (block_ids, af_indices, af_offsets) = coord_block(
                    coords.into_iter(),
                    &coord_bounds,
                    PER_BLOCK,
                    ndim,
                    num_coords,
                );

                Box::pin(async move {
                    let mut start = 0.0f64;
                    let mut values = vec![];
                    for block_id in block_ids {
                        debug!("block {} starts at {}", block_id, start);

                        let (block_offsets, new_start) =
                            block_offsets(&af_indices, &af_offsets, start, block_id);

                        debug!("reading {} block_offsets", block_offsets.elements());
                        let block = file_clone.read_block(txn_id, block_id.into()).await?;
                        values.extend(block.get(&block_offsets.into()).to_vec());
                        start = new_start;
                    }

                    Ok(Array::from(values))
                })
            });

        let blocks: TCTryStream<Array> = Box::pin(values);
        Box::pin(future::ready(Ok(blocks)))
    }

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        let bounds = self.rebase.invert_bounds(bounds);
        self.source.slice(bounds)
    }

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Transpose> {
        BlockListTranspose::new(self, permutation)
    }

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, number: Number) -> TCResult<()> {
        self.shape().validate_bounds(&bounds)?;

        let bounds = self.rebase.invert_bounds(bounds);
        self.source.write_value(txn_id, bounds, number).await
    }

    fn write_value_at(
        &'_ self,
        txn_id: TxnId,
        coord: Coord,
        value: Number,
    ) -> TCBoxTryFuture<'_, ()> {
        Box::pin(async move {
            self.shape().validate_coord(&coord)?;
            let coord = self.rebase.invert_coord(&coord);
            self.source.write_value_at(txn_id, coord, value).await
        })
    }
}

impl<FD, FS, D, T> ReadValueAt<D> for BlockListFileSlice<FD, FS, D, T>
where
    FD: File<Array> + TryFrom<D::File, Error = TCError>,
    FS: File<Node> + TryFrom<D::File, Error = TCError>,
    D: Dir,
    T: Transaction<D>,
    D::FileClass: From<TensorType>,
{
    type Txn = T;

    fn read_value_at<'a>(self, txn: Self::Txn, coord: Coord) -> Read<'a> {
        Box::pin(async move {
            self.shape().validate_coord(&coord)?;
            let coord = self.rebase.invert_coord(&coord);
            self.source.read_value_at(txn, coord).await
        })
    }
}

#[inline]
fn div_ceil(l: u64, r: u64) -> u64 {
    if l % r == 0 {
        l / r
    } else {
        (l / r) + 1
    }
}
