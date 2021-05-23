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
use number_general::{Number, NumberInstance, NumberType};
use strided::Stride;

use tc_error::*;
use tc_transact::fs::{BlockData, BlockId, Dir, File};
use tc_transact::{Transaction, TxnId};
use tc_value::Value;
use tcgeneric::{TCBoxTryFuture, TCTryStream};

use crate::{Bounds, Coord, Read, ReadValueAt, Schema, Shape, TensorAccess};

use super::{block_offsets, coord_block, DenseAccess, DenseAccessor, PER_BLOCK};

#[derive(Clone)]
pub struct BlockListFile<F, D, T> {
    file: F,
    dtype: NumberType,
    shape: Shape,
    dir: PhantomData<D>,
    txn: PhantomData<T>,
}

impl<F: File<Array>, D: Dir, T: Transaction<D>> BlockListFile<F, D, T> {
    pub async fn constant(file: F, txn_id: TxnId, shape: Shape, value: Number) -> TCResult<Self> {
        let size = shape.size();
        let per_block = Array::max_size();

        let value_clone = value.clone();
        let blocks = (0..(size / per_block))
            .map(move |_| Ok(Array::constant(value_clone.clone(), per_block as usize)));

        let trailing_len = (size % per_block) as usize;
        if trailing_len > 0 {
            let blocks = blocks.chain(iter::once(Ok(Array::constant(value.clone(), trailing_len))));
            BlockListFile::from_blocks(file, txn_id, shape, value.class(), stream::iter(blocks))
                .await
        } else {
            BlockListFile::from_blocks(file, txn_id, shape, value.class(), stream::iter(blocks))
                .await
        }
    }

    pub async fn from_blocks<S: Stream<Item = TCResult<Array>> + Send + Unpin>(
        file: F,
        txn_id: TxnId,
        shape: Shape,
        dtype: NumberType,
        blocks: S,
    ) -> TCResult<Self> {
        blocks
            .enumerate()
            .map(|(i, r)| r.map(|block| (BlockId::from(i), block)))
            .map_ok(|(id, block)| file.create_block(txn_id, id, block))
            .try_buffer_unordered(num_cpus::get())
            .try_fold((), |_, _| future::ready(Ok(())))
            .await?;

        Ok(BlockListFile {
            dtype,
            shape,
            file,
            dir: PhantomData,
            txn: PhantomData,
        })
    }

    pub async fn from_values<S: Stream<Item = Number> + Send + Unpin>(
        file: F,
        txn_id: TxnId,
        shape: Shape,
        dtype: NumberType,
        values: S,
    ) -> TCResult<Self> {
        let mut i = 0u64;
        let mut values = values.chunks(Array::max_size() as usize);
        while let Some(chunk) = values.next().await {
            let block_id = BlockId::from(i);
            let block = Array::from(chunk).cast_into(dtype);
            file.create_block(txn_id, block_id, block).await?;
            i += 1;
        }

        Ok(BlockListFile {
            dtype,
            shape,
            file,
            dir: PhantomData,
            txn: PhantomData,
        })
    }

    pub fn into_stream(self, txn_id: TxnId) -> impl Stream<Item = TCResult<Array>> + Unpin {
        let num_blocks = div_ceil(self.size(), Array::max_size());
        let blocks = stream::iter((0..num_blocks).into_iter().map(BlockId::from))
            .then(move |block_id| self.file.clone().read_block_owned(txn_id, block_id))
            .map_ok(|block| (*block).clone());

        Box::pin(blocks)
    }

    pub async fn merge_sort(&self, txn_id: TxnId) -> TCResult<()> {
        let num_blocks = div_ceil(self.size(), Array::max_size());
        if num_blocks == 1 {
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

            let (left_sorted, right_sorted) = block.split(Array::max_size() as usize)?;
            *left = left_sorted;
            *right = right_sorted;
        }

        Ok(())
    }
}

impl<F: Send, D: Send, T: Send> TensorAccess for BlockListFile<F, D, T> {
    fn dtype(&self) -> NumberType {
        self.dtype
    }

    fn ndim(&self) -> usize {
        self.shape.len()
    }

    fn shape(&'_ self) -> &'_ Shape {
        &self.shape
    }

    fn size(&self) -> u64 {
        self.shape.size()
    }
}

#[async_trait]
impl<F: File<Array>, D: Dir, T: Transaction<D>> DenseAccess<F, D, T> for BlockListFile<F, D, T> {
    fn accessor(self) -> DenseAccessor<F, D, T> {
        DenseAccessor::File(self)
    }

    fn block_stream<'a>(self, txn: T) -> TCBoxTryFuture<'a, TCTryStream<'a, Array>> {
        Box::pin(async move {
            let size = self.size();
            let file = self.file;
            let block_stream = Box::pin(
                stream::iter(0..(div_ceil(size, Array::max_size())))
                    .map(BlockId::from)
                    .then(move |block_id| file.clone().read_block_owned(*txn.id(), block_id))
                    .map_ok(|block| (*block).clone()),
            );

            let block_stream: TCTryStream<Array> = Box::pin(block_stream);
            Ok(block_stream)
        })
    }

    async fn write_value(&self, txn_id: TxnId, bounds: Bounds, value: Number) -> TCResult<()> {
        debug!("BlockListFile::write_value {} at {}", value, bounds);

        if !self.shape().contains_bounds(&bounds) {
            return Err(TCError::bad_request("Bounds out of bounds", bounds));
        } else if bounds.len() == self.ndim() {
            if let Some(coord) = bounds.as_coord() {
                return self.write_value_at(txn_id, coord, value).await;
            }
        }

        let bounds = self.shape().slice_bounds(bounds);
        let coord_bounds = coord_bounds(self.shape());

        let per_block = Array::max_size() as usize;
        stream::iter(bounds.affected())
            .chunks(per_block)
            .map(|coords| {
                let ndim = coords[0].len();
                let num_coords = coords.len() as u64;
                let (block_ids, af_indices, af_offsets) = coord_block(
                    coords.into_iter(),
                    &coord_bounds,
                    per_block,
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

            let value = value.into_type(self.dtype);

            let offset: u64 = coord_bounds(self.shape())
                .iter()
                .zip(coord.iter())
                .map(|(d, x)| d * x)
                .sum();

            let block_id = BlockId::from(offset / Array::max_size());
            let mut block = self.file.write_block(txn_id, block_id).await?;

            (*block)
                .set_value((offset / Array::max_size()) as usize, value)
                .map_err(TCError::from)
        })
    }
}

impl<F: File<Array>, D: Dir, T: Transaction<D>> ReadValueAt<D, T> for BlockListFile<F, D, T> {
    fn read_value_at<'a>(&'a self, txn: &'a T, coord: Coord) -> Read<'a> {
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

            let block_id = BlockId::from(offset / Array::max_size());
            let block = self.file.read_block(*txn.id(), block_id).await?;

            debug!(
                "read offset {} from block of length {}",
                offset % Array::max_size(),
                block.len()
            );

            let value = block.get_value((offset % Array::max_size()) as usize);

            Ok((coord, value))
        })
    }
}

#[async_trait]
impl<F: File<Array>, D: Send, T: Send> de::FromStream for BlockListFile<F, D, T> {
    type Context = (TxnId, F, Schema);

    async fn from_stream<De: de::Decoder>(
        cxt: (TxnId, F, Schema),
        decoder: &mut De,
    ) -> Result<Self, De::Error> {
        let (txn_id, file, (dtype, shape)) = cxt;
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
            Ok(Self {
                file,
                shape,
                dtype,
                dir: PhantomData,
                txn: PhantomData,
            })
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
        let mut buf = [T::default(); BUF_SIZE];
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
        debug_assert_eq!(Array::max_size(), PER_BLOCK as u64);
        self.visit_array::<bool, A, PER_BLOCK>(access).await
    }

    async fn visit_array_i16<A: de::ArrayAccess<i16>>(
        self,
        access: A,
    ) -> Result<Self::Value, A::Error> {
        const BUF_SIZE: usize = PER_BLOCK / 2;
        debug_assert_eq!(Array::max_size() / 2, BUF_SIZE as u64);
        self.visit_array::<i16, A, BUF_SIZE>(access).await
    }

    async fn visit_array_i32<A: de::ArrayAccess<i32>>(
        self,
        access: A,
    ) -> Result<Self::Value, A::Error> {
        const BUF_SIZE: usize = PER_BLOCK / 4;
        debug_assert_eq!(Array::max_size() / 4, BUF_SIZE as u64);
        self.visit_array::<i32, A, BUF_SIZE>(access).await
    }

    async fn visit_array_i64<A: de::ArrayAccess<i64>>(
        self,
        access: A,
    ) -> Result<Self::Value, A::Error> {
        const BUF_SIZE: usize = PER_BLOCK / 8;
        debug_assert_eq!(Array::max_size() / 8, BUF_SIZE as u64);
        self.visit_array::<i64, A, BUF_SIZE>(access).await
    }

    async fn visit_array_u8<A: de::ArrayAccess<u8>>(
        self,
        access: A,
    ) -> Result<Self::Value, A::Error> {
        debug_assert_eq!(Array::max_size(), PER_BLOCK as u64);
        self.visit_array::<u8, A, PER_BLOCK>(access).await
    }

    async fn visit_array_u16<A: de::ArrayAccess<u16>>(
        self,
        access: A,
    ) -> Result<Self::Value, A::Error> {
        const BUF_SIZE: usize = PER_BLOCK / 2;
        debug_assert_eq!(Array::max_size() / 2, BUF_SIZE as u64);
        self.visit_array::<u16, A, BUF_SIZE>(access).await
    }

    async fn visit_array_u32<A: de::ArrayAccess<u32>>(
        self,
        access: A,
    ) -> Result<Self::Value, A::Error> {
        const BUF_SIZE: usize = PER_BLOCK / 4;
        debug_assert_eq!(Array::max_size() / 4, BUF_SIZE as u64);
        self.visit_array::<u32, A, BUF_SIZE>(access).await
    }

    async fn visit_array_u64<A: de::ArrayAccess<u64>>(
        self,
        access: A,
    ) -> Result<Self::Value, A::Error> {
        const BUF_SIZE: usize = PER_BLOCK / 8;
        debug_assert_eq!(Array::max_size() / 8, BUF_SIZE as u64);
        self.visit_array::<u64, A, BUF_SIZE>(access).await
    }

    async fn visit_array_f32<A: de::ArrayAccess<f32>>(
        self,
        access: A,
    ) -> Result<Self::Value, A::Error> {
        const BUF_SIZE: usize = PER_BLOCK / 4;
        debug_assert_eq!(Array::max_size() / 4, BUF_SIZE as u64);
        self.visit_array::<f32, A, BUF_SIZE>(access).await
    }

    async fn visit_array_f64<A: de::ArrayAccess<f64>>(
        self,
        access: A,
    ) -> Result<Self::Value, A::Error> {
        const BUF_SIZE: usize = PER_BLOCK / 8;
        debug_assert_eq!(Array::max_size() / 8, BUF_SIZE as u64);
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
        const BUF_SIZE: usize = (PER_BLOCK / 4) * 2;
        debug_assert_eq!((Array::max_size() / 4) * 2, BUF_SIZE as u64);
        self.visit_array::<_, f32, A, BUF_SIZE>(access).await
    }

    async fn visit_array_f64<A: de::ArrayAccess<f64>>(
        self,
        access: A,
    ) -> Result<Self::Value, A::Error> {
        const BUF_SIZE: usize = (PER_BLOCK / 8) * 2;
        debug_assert_eq!((Array::max_size() / 8) * 2, BUF_SIZE as u64);
        self.visit_array::<_, f64, A, BUF_SIZE>(access).await
    }
}

#[inline]
fn coord_bounds(shape: &Shape) -> Coord {
    (0..shape.len())
        .map(|axis| shape[axis + 1..].iter().product())
        .collect()
}

#[inline]
fn div_ceil(l: u64, r: u64) -> u64 {
    if l % r == 0 {
        l / r
    } else {
        (l / r) + 1
    }
}
