use std::convert::{TryFrom, TryInto};
use std::iter;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use arrayfire as af;
use async_trait::async_trait;
use futures::future;
use futures::stream::{self, Stream, StreamExt, TryStreamExt};
use futures::try_join;
use num::integer::div_ceil;

use crate::block::BlockId;
use crate::block::File;
use crate::class::{Instance, TCBoxTryFuture, TCResult, TCTryStream};
use crate::collection::tensor::bounds::*;
use crate::collection::tensor::class::TensorInstance;
use crate::error;
use crate::scalar::number::*;
use crate::transaction::{Transact, Txn, TxnId};

use super::array::Array;
use super::BlockList;

const ERR_CORRUPT: &str = "DenseTensor corrupted! Please file a bug report.";
pub const PER_BLOCK: usize = 131_072; // = 1 mibibyte / 64 bits

#[derive(Clone)]
pub struct BlockListFile {
    file: Arc<File<Array>>,
    dtype: NumberType,
    shape: Shape,
    coord_bounds: Vec<u64>,
}

impl BlockListFile {
    pub async fn constant(txn: &Txn, shape: Shape, value: Number) -> TCResult<BlockListFile> {
        let size = shape.size();

        let value_clone = value.clone();
        let blocks = (0..(size / PER_BLOCK as u64))
            .map(move |_| Ok(Array::constant(value_clone.clone(), PER_BLOCK)));
        let trailing_len = (size % (PER_BLOCK as u64)) as usize;
        if trailing_len > 0 {
            let blocks = blocks.chain(iter::once(Ok(Array::constant(value.clone(), trailing_len))));
            BlockListFile::from_blocks(txn, shape, value.class(), stream::iter(blocks)).await
        } else {
            BlockListFile::from_blocks(txn, shape, value.class(), stream::iter(blocks)).await
        }
    }

    pub async fn from_blocks<S: Stream<Item = TCResult<Array>> + Send + Unpin>(
        txn: &Txn,
        shape: Shape,
        dtype: NumberType,
        blocks: S,
    ) -> TCResult<BlockListFile> {
        let file = txn.context().await?;

        blocks
            .enumerate()
            .map(|(i, r)| r.map(|block| (BlockId::from(i), block)))
            .map_ok(|(id, block)| file.clone().create_block(txn.id().clone(), id, block))
            .try_buffer_unordered(2)
            .try_fold((), |_, _| future::ready(Ok(())))
            .await?;

        let coord_bounds = (0..shape.len())
            .map(|axis| shape[axis + 1..].iter().product())
            .collect();

        Ok(BlockListFile {
            dtype,
            shape,
            file,
            coord_bounds,
        })
    }

    pub async fn from_values<S: Stream<Item = Number> + Send + Unpin>(
        txn: &Txn,
        shape: Shape,
        dtype: NumberType,
        values: S,
    ) -> TCResult<BlockListFile> {
        let file = txn.context().await?;

        let mut i = 0u64;
        let mut values = values.chunks(PER_BLOCK);
        while let Some(chunk) = values.next().await {
            let block_id = BlockId::from(i);
            let block = Array::cast_from_values(chunk, dtype)?;
            file.clone()
                .create_block(txn.id().clone(), block_id, block)
                .await?;

            i += 1;
        }

        let coord_bounds = (0..shape.len())
            .map(|axis| shape[axis + 1..].iter().product())
            .collect();

        Ok(BlockListFile {
            dtype,
            shape,
            file,
            coord_bounds,
        })
    }

    async fn merge_sort(&self, txn_id: &TxnId) -> TCResult<()> {
        let num_blocks = div_ceil(self.size(), PER_BLOCK as u64);
        if num_blocks == 1 {
            let block_id = BlockId::from(0u64);
            let mut block = self
                .file
                .get_block(txn_id, &block_id)
                .await?
                .upgrade()
                .await?;
            block.sort();
            return Ok(());
        }

        for block_id in 0..(num_blocks - 1) {
            let next_block_id = BlockId::from(block_id + 1);
            let block_id = BlockId::from(block_id);

            let left = self.file.get_block(txn_id, &block_id);
            let right = self.file.get_block(txn_id, &next_block_id);
            let (left, right) = try_join!(left, right)?;
            let (mut left, mut right) = try_join!(left.upgrade(), right.upgrade())?;

            let mut block = Array::concatenate(&left, &right)?;
            block.sort();

            let (left_sorted, right_sorted) = block.split(PER_BLOCK)?;
            *left = left_sorted;
            *right = right_sorted;
        }

        Ok(())
    }
}

impl TensorInstance for BlockListFile {
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
impl BlockList for BlockListFile {
    fn block_stream<'a>(self: Arc<Self>, txn: Txn) -> TCBoxTryFuture<'a, TCTryStream<Array>> {
        Box::pin(async move {
            let block_stream = Box::pin(
                stream::iter(0..(div_ceil(self.size(), PER_BLOCK as u64)))
                    .map(BlockId::from)
                    .then(move |block_id| {
                        self.file
                            .clone()
                            .get_block_owned(txn.id().clone(), block_id)
                    }),
            );

            let block_stream =
                block_stream.and_then(|block| future::ready(Ok(block.deref().clone())));

            let block_stream: TCTryStream<Array> = Box::pin(block_stream);

            Ok(block_stream)
        })
    }

    async fn value_stream_slice(
        self: Arc<Self>,
        txn: Txn,
        bounds: Bounds,
    ) -> TCResult<TCTryStream<Number>> {
        if bounds.deref() == self.shape().all().deref() {
            return self.value_stream(txn).await;
        }

        if !self.shape.contains_bounds(&bounds) {
            return Err(error::bad_request("Invalid bounds", bounds));
        }

        let ndim = bounds.ndim(self.shape());

        let coord_bounds = af::Array::new(
            &self.coord_bounds,
            af::Dim4::new(&[self.ndim() as u64, 1, 1, 1]),
        );

        let selected = stream::iter(bounds.affected())
            .chunks(PER_BLOCK)
            .then(move |coords| {
                let (block_ids, af_indices, af_offsets, num_coords) =
                    coord_block(coords.into_iter(), &coord_bounds, PER_BLOCK, ndim);

                let this = self.clone();
                let txn = txn.clone();

                Box::pin(async move {
                    let mut start = 0.0f64;
                    let mut values = vec![];
                    for block_id in block_ids {
                        let (block_offsets, new_start) =
                            block_offsets(&af_indices, &af_offsets, num_coords, start, block_id);

                        match this
                            .file
                            .clone()
                            .get_block(txn.id(), &block_id.into())
                            .await
                        {
                            Ok(block) => {
                                let array: &Array = block.deref().try_into().unwrap();
                                values.extend(array.get(block_offsets));
                            }
                            Err(cause) => return stream::iter(vec![Err(cause)]),
                        }

                        start = new_start;
                    }

                    let values: Vec<TCResult<Number>> = values.into_iter().map(Ok).collect();
                    stream::iter(values)
                })
            });

        let selected: TCTryStream<Number> = Box::pin(selected.flatten());
        Ok(selected)
    }

    async fn read_value_at(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        if !self.shape().contains_coord(coord) {
            let coord: Vec<String> = coord.iter().map(|c| c.to_string()).collect();
            return Err(error::bad_request(
                "Coordinate is out of bounds",
                coord.join(", "),
            ));
        }

        let offset: u64 = self
            .coord_bounds
            .iter()
            .zip(coord.iter())
            .map(|(d, x)| d * x)
            .sum();
        let block_id = BlockId::from(offset / PER_BLOCK as u64);
        let block = self.file.get_block(txn.id(), &block_id).await?;
        block
            .deref()
            .get_value((offset % PER_BLOCK as u64) as usize)
            .ok_or_else(|| error::internal(ERR_CORRUPT))
    }

    async fn write_value(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
        value: Number,
    ) -> TCResult<()> {
        if !self.shape().contains_bounds(&bounds) {
            return Err(error::bad_request("Bounds out of bounds", bounds));
        }

        let ndim = bounds.ndim(self.shape());

        let coord_bounds = af::Array::new(
            &self.coord_bounds,
            af::Dim4::new(&[self.ndim() as u64, 1, 1, 1]),
        );

        stream::iter(bounds.affected())
            .chunks(PER_BLOCK)
            .map(|coords| {
                let (block_ids, af_indices, af_offsets, num_coords) =
                    coord_block(coords.into_iter(), &coord_bounds, PER_BLOCK, ndim);

                let this = self.clone();
                let value = value.clone();
                let txn_id = txn_id;

                Ok(async move {
                    let mut start = 0.0f64;
                    for block_id in block_ids {
                        let value = value.clone();
                        let (block_offsets, new_start) =
                            block_offsets(&af_indices, &af_offsets, num_coords, start, block_id);

                        let block_id = BlockId::from(block_id);
                        let mut block = this
                            .file
                            .get_block(&txn_id, &block_id)
                            .await?
                            .upgrade()
                            .await?;

                        let value = Array::constant(value, (new_start - start) as usize);
                        block.deref_mut().set(block_offsets, &value)?;
                        start = new_start;
                    }

                    Ok(())
                })
            })
            .try_buffer_unordered(2)
            .fold(Ok(()), |_, r| future::ready(r))
            .await
    }

    fn write_value_at(&self, txn_id: TxnId, coord: Vec<u64>, value: Number) -> TCBoxTryFuture<()> {
        Box::pin(async move {
            if !self.shape().contains_coord(&coord) {
                return Err(error::bad_request(
                    "Invalid coordinate",
                    format!("[{:?}]", coord),
                ));
            }

            let value = value.into_type(self.dtype);

            let offset: u64 = self
                .coord_bounds
                .iter()
                .zip(coord.iter())
                .map(|(d, x)| d * x)
                .sum();

            let block_id = BlockId::from(offset / PER_BLOCK as u64);

            let mut block = self
                .file
                .get_block(&txn_id, &block_id)
                .await?
                .upgrade()
                .await?;

            block
                .deref_mut()
                .set_value((offset % PER_BLOCK as u64) as usize, value)
        })
    }
}

#[async_trait]
impl Transact for BlockListFile {
    async fn commit(&self, txn_id: &TxnId) {
        self.file.commit(txn_id).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.file.rollback(txn_id).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.file.finalize(txn_id).await
    }
}

pub async fn sort_coords<S: Stream<Item = TCResult<Vec<u64>>> + Send + Unpin + 'static>(
    txn: Txn,
    coords: S,
    num_coords: u64,
    shape: &Shape,
) -> TCResult<impl Stream<Item = TCResult<Vec<u64>>>> {
    let ndim = shape.len();
    let coord_bounds: Vec<u64> = (0..shape.len())
        .map(|axis| shape[axis + 1..].iter().product())
        .collect();
    let coord_bounds: af::Array<u64> =
        af::Array::new(&coord_bounds, af::Dim4::new(&[ndim as u64, 1, 1, 1]));
    let coord_bounds_copy = coord_bounds.copy();
    let shape: af::Array<u64> =
        af::Array::new(&shape.to_vec(), af::Dim4::new(&[ndim as u64, 1, 1, 1]));

    let blocks = coords
        .chunks(PER_BLOCK)
        .map(|block| block.into_iter().collect::<TCResult<Vec<Vec<u64>>>>())
        .map_ok(move |block| {
            let num_coords = block.len();
            let block = block.into_iter().flatten().collect::<Vec<u64>>();
            af::Array::new(
                &block,
                af::Dim4::new(&[ndim as u64, num_coords as u64, 1, 1]),
            )
        })
        .map_ok(move |block| af::sum(&(block * coord_bounds.copy()), 1))
        .and_then(|block| future::ready(Array::try_from(block)));

    let blocks: TCTryStream<Array> = Box::pin(blocks);
    let block_list = BlockListFile::from_blocks(
        &txn,
        Shape::from(vec![num_coords]),
        NumberType::uint64(),
        blocks,
    )
    .await
    .map(Arc::new)?;
    block_list.merge_sort(txn.id()).await?;

    let coords = block_list
        .block_stream(txn)
        .await?
        .map_ok(|block| block.into_af_array::<u64>())
        .map_ok(|block| af::moddims(&block, af::Dim4::new(&[1, block.elements() as u64, 1, 1])))
        .map_ok(move |block| {
            let dims = af::Dim4::new(&[ndim as u64, block.elements() as u64, 1, 1]);
            let block = af::tile(&block, dims);
            let coord_bounds = af::tile(&coord_bounds_copy, dims);
            let shape = af::tile(&shape, dims);
            (block / coord_bounds) % shape
        })
        .map_ok(|coord_block| {
            let mut coords: Vec<u64> = Vec::with_capacity(coord_block.elements());
            coord_block.host(&mut coords);
            coords
        })
        .map_ok(move |coords| {
            stream::iter(coords.into_iter())
                .chunks(ndim)
                .map(Result::<Vec<u64>, error::TCError>::Ok)
        })
        .try_flatten();

    Ok(coords)
}

fn block_offsets(
    af_indices: &af::Array<u64>,
    af_offsets: &af::Array<u64>,
    num_coords: u64,
    start: f64,
    block_id: u64,
) -> (af::Array<u64>, f64) {
    let num_to_update = af::sum_all(&af::eq(
        af_indices,
        &af::constant(block_id, af_indices.dims()),
        false,
    ))
    .0;
    let block_offsets = af::index(
        af_offsets,
        &[
            af::Seq::new(block_id as f64, block_id as f64, 1.0f64),
            af::Seq::new(start, (start + num_to_update) - 1.0f64, 1.0f64),
        ],
    );
    let block_offsets = af::moddims(&block_offsets, af::Dim4::new(&[num_coords as u64, 1, 1, 1]));

    (block_offsets, (start + num_to_update))
}

fn coord_block<I: Iterator<Item = Vec<u64>>>(
    coords: I,
    coord_bounds: &af::Array<u64>,
    per_block: usize,
    ndim: usize,
) -> (Vec<u64>, af::Array<u64>, af::Array<u64>, u64) {
    let coords: Vec<u64> = coords.flatten().collect();
    let num_coords = coords.len() / ndim;
    let af_coords_dim = af::Dim4::new(&[num_coords as u64, ndim as u64, 1, 1]);
    let af_coords = af::Array::new(&coords, af_coords_dim) * af::tile(coord_bounds, af_coords_dim);
    let af_coords = af::sum(&af_coords, 1);
    let af_per_block = af::constant(
        per_block as u64,
        af::Dim4::new(&[1, num_coords as u64, 1, 1]),
    );
    let af_offsets = af_coords.copy() % af_per_block.copy();
    let af_indices = af_coords / af_per_block;
    let af_block_ids = af::set_unique(&af_indices, true);

    let mut block_ids: Vec<u64> = Vec::with_capacity(af_block_ids.elements());
    af_block_ids.host(&mut block_ids);
    (block_ids, af_indices, af_offsets, num_coords as u64)
}
