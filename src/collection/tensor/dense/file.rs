use std::convert::{TryFrom, TryInto};
use std::iter;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use arrayfire as af;
use async_trait::async_trait;
use futures::future;
use futures::stream::{self, Stream, StreamExt, TryStreamExt};
use futures::try_join;
use log::debug;
use num::integer::div_ceil;

use crate::block::BlockId;
use crate::block::File;
use crate::class::Instance;
use crate::error;
use crate::general::{TCBoxTryFuture, TCResult, TCTryStreamOld};
use crate::scalar::number::*;
use crate::transaction::{Transact, Txn, TxnId};

use super::super::bounds::*;
use super::super::TensorAccessor;

use super::array::Array;
use super::BlockList;

pub const PER_BLOCK: usize = 131_072; // = 1 mibibyte / 64 bits

#[derive(Clone)]
pub struct BlockListFile {
    file: Arc<File<Array>>,
    dtype: NumberType,
    shape: Shape,
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

        Ok(BlockListFile { dtype, shape, file })
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

        Ok(BlockListFile { dtype, shape, file })
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

impl TensorAccessor for BlockListFile {
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
    fn block_stream<'a>(self: Arc<Self>, txn: Txn) -> TCBoxTryFuture<'a, TCTryStreamOld<Array>> {
        Box::pin(async move {
            let file = self.file.clone();
            let block_stream = Box::pin(
                stream::iter(0..(div_ceil(self.size(), PER_BLOCK as u64)))
                    .map(BlockId::from)
                    .then(move |block_id| file.clone().get_block_owned(txn.id().clone(), block_id)),
            );

            let block_stream =
                block_stream.and_then(|block| future::ready(Ok(block.deref().clone())));

            let block_stream: TCTryStreamOld<Array> = Box::pin(block_stream);

            Ok(block_stream)
        })
    }

    async fn value_stream_slice(
        self: Arc<Self>,
        txn: Txn,
        bounds: Bounds,
    ) -> TCResult<TCTryStreamOld<Number>> {
        if bounds == Bounds::all(self.shape()) {
            return self.value_stream(txn).await;
        }

        if !self.shape.contains_bounds(&bounds) {
            return Err(error::bad_request("Invalid bounds", bounds));
        }

        let bounds = self.shape().slice_bounds(bounds);
        let coord_bounds = coord_bounds(self.shape());

        let selected = stream::iter(bounds.affected())
            .chunks(PER_BLOCK)
            .then(move |coords| {
                let ndim = coords[0].len();
                let num_coords = coords.len() as u64;
                let (block_ids, af_indices, af_offsets) = coord_block(
                    coords.into_iter(),
                    &coord_bounds,
                    PER_BLOCK,
                    ndim,
                    num_coords,
                );

                let this = self.clone();
                let txn = txn.clone();

                Box::pin(async move {
                    let mut start = 0.0f64;
                    let mut values = vec![];
                    for block_id in block_ids {
                        debug!("block {} starts at {}", block_id, start);

                        let (block_offsets, new_start) =
                            block_offsets(&af_indices, &af_offsets, start, block_id);

                        match this
                            .file
                            .clone()
                            .get_block(txn.id(), &block_id.into())
                            .await
                        {
                            Ok(block) => {
                                let array: &Array = block.deref();
                                values.extend(array.get(block_offsets).into_values());
                            }
                            Err(cause) => return stream::iter(vec![Err(cause)]),
                        }

                        start = new_start;
                    }

                    let values: Vec<TCResult<Number>> = values.into_iter().map(Ok).collect();
                    stream::iter(values)
                })
            });

        let selected: TCTryStreamOld<Number> = Box::pin(selected.flatten());
        Ok(selected)
    }

    async fn read_value_at(&self, txn: &Txn, coord: &[u64]) -> TCResult<Number> {
        debug!(
            "read value at {:?} from BlockListFile with shape {}",
            coord,
            self.shape()
        );

        if !self.shape().contains_coord(coord) {
            let coord: Vec<String> = coord.iter().map(|c| c.to_string()).collect();
            return Err(error::bad_request(
                "Coordinate is out of bounds",
                coord.join(", "),
            ));
        }

        let offset: u64 = coord_bounds(self.shape())
            .iter()
            .zip(coord.iter())
            .map(|(d, x)| d * x)
            .sum();
        debug!("coord {:?} is offset {}", coord, offset);

        let block_id = BlockId::from(offset / PER_BLOCK as u64);
        let block = self.file.get_block(txn.id(), &block_id).await?;

        debug!(
            "read offset {} at block {} (length {})",
            (offset % PER_BLOCK as u64),
            block_id,
            block.len()
        );
        let value = block.get_value((offset % PER_BLOCK as u64) as usize);

        Ok(value)
    }

    async fn write_value(
        self: Arc<Self>,
        txn_id: TxnId,
        bounds: Bounds,
        value: Number,
    ) -> TCResult<()> {
        debug!("BlockListFile::write_value {} at {}", value, bounds);

        if !self.shape().contains_bounds(&bounds) {
            return Err(error::bad_request("Bounds out of bounds", bounds));
        } else if bounds.len() == self.ndim() && bounds.is_coord() {
            return self.write_value_at(txn_id, bounds.try_into()?, value).await;
        }

        let bounds = self.shape().slice_bounds(bounds);
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

                let this = self.clone();
                let value = value.clone();
                let txn_id = txn_id;

                Ok(async move {
                    let mut start = 0.0f64;
                    for block_id in block_ids {
                        let value = value.clone();
                        let (block_offsets, new_start) =
                            block_offsets(&af_indices, &af_offsets, start, block_id);

                        let block_id = BlockId::from(block_id);
                        let mut block = this
                            .file
                            .get_block(&txn_id, &block_id)
                            .await?
                            .upgrade()
                            .await?;

                        debug!("write {} to", value);
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

            let offset: u64 = coord_bounds(self.shape())
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

pub async fn sort_coords<'a, S: Stream<Item = TCResult<Vec<u64>>> + Send + Unpin + 'a>(
    txn: Txn,
    coords: S,
    num_coords: u64,
    shape: &Shape,
) -> TCResult<impl Stream<Item = TCResult<Vec<u64>>> + 'a> {
    let blocks =
        coords_to_offsets(shape, coords).and_then(|block| future::ready(Array::try_from(block)));

    let block_list = BlockListFile::from_blocks(
        &txn,
        Shape::from(vec![num_coords]),
        UIntType::U64.into(),
        Box::pin(blocks),
    )
    .await?;

    block_list.merge_sort(txn.id()).await?;

    let blocks = Arc::new(block_list).block_stream(txn).await?;
    Ok(offsets_to_coords(shape, blocks))
}

fn coords_to_offsets<S: Stream<Item = TCResult<Vec<u64>>>>(
    shape: &Shape,
    coords: S,
) -> impl Stream<Item = TCResult<af::Array<u64>>> {
    let ndim = shape.len() as u64;
    let coord_bounds = coord_bounds(shape);
    let af_coord_bounds: af::Array<u64> =
        af::Array::new(&coord_bounds, af::Dim4::new(&[ndim, 1, 1, 1]));

    coords
        .chunks(PER_BLOCK)
        .map(|block| block.into_iter().collect::<TCResult<Vec<Vec<u64>>>>())
        .map_ok(move |block| {
            let num_coords = block.len();
            let block = block.into_iter().flatten().collect::<Vec<u64>>();
            af::Array::new(&block, af::Dim4::new(&[ndim, num_coords as u64, 1, 1]))
        })
        .map_ok(move |block| {
            let offsets = af::mul(&block, &af_coord_bounds, true);
            af::sum(&offsets, 0)
        })
        .map_ok(|block| af::moddims(&block, af::Dim4::new(&[block.elements() as u64, 1, 1, 1])))
}

fn offsets_to_coords<S: Stream<Item = TCResult<Array>>>(
    shape: &Shape,
    blocks: S,
) -> impl Stream<Item = TCResult<Vec<u64>>> {
    let ndim = shape.len() as u64;
    let coord_bounds = coord_bounds(shape);
    let af_coord_bounds: af::Array<u64> =
        af::Array::new(&coord_bounds, af::Dim4::new(&[1, ndim, 1, 1]));
    let af_shape: af::Array<u64> = af::Array::new(&shape.to_vec(), af::Dim4::new(&[1, ndim, 1, 1]));
    let ndim = shape.len();

    blocks
        .map_ok(|block| block.into_af_array::<u64>())
        .map_ok(move |block| {
            let offsets = af::div(&block, &af_coord_bounds, true);
            af::modulo(&offsets, &af_shape, true)
        })
        .map_ok(|coord_block| {
            let mut coords = vec![0u64; coord_block.elements()];
            af::transpose(&coord_block, false).host(&mut coords);
            coords
        })
        .map_ok(move |coords| {
            stream::iter(coords.into_iter())
                .chunks(ndim)
                .map(TCResult::<Vec<u64>>::Ok)
        })
        .try_flatten()
}

fn coord_bounds(shape: &Shape) -> Vec<u64> {
    (0..shape.len())
        .map(|axis| shape[axis + 1..].iter().product())
        .collect()
}

fn block_offsets(
    af_indices: &af::Array<u64>,
    af_offsets: &af::Array<u64>,
    start: f64,
    block_id: u64,
) -> (af::Array<u64>, f64) {
    assert_eq!(af_indices.elements(), af_offsets.elements());

    let num_to_update = af::sum_all(&af::eq(
        af_indices,
        &af::constant(block_id, af::Dim4::new(&[1, 1, 1, 1])),
        true,
    ))
    .0;

    if num_to_update == 0f64 {
        return (af::Array::new_empty(af::Dim4::default()), start);
    }

    assert!((start + num_to_update) as usize <= af_offsets.elements());

    let block_offsets = af::index(
        af_offsets,
        &[af::Seq::new(start, (start + num_to_update) - 1f64, 1f64)],
    );

    (block_offsets, (start + num_to_update))
}

fn coord_block<I: Iterator<Item = Vec<u64>>>(
    coords: I,
    coord_bounds: &[u64],
    per_block: usize,
    ndim: usize,
    num_coords: u64,
) -> (Vec<u64>, af::Array<u64>, af::Array<u64>) {
    let coords: Vec<u64> = coords.flatten().collect();
    assert!(coords.len() > 0);
    assert!(ndim > 0);

    let af_per_block = af::constant(per_block as u64, af::Dim4::new(&[1, 1, 1, 1]));
    let af_coord_bounds = af::Array::new(coord_bounds, af::Dim4::new(&[ndim as u64, 1, 1, 1]));

    let af_coords = af::Array::new(
        &coords,
        af::Dim4::new(&[ndim as u64, num_coords as u64, 1, 1]),
    );
    let af_coords = af::mul(&af_coords, &af_coord_bounds, true);
    let af_coords = af::sum(&af_coords, 0);

    let af_offsets = af::modulo(&af_coords, &af_per_block, true);
    let af_indices = af_coords / af_per_block;

    let af_block_ids = af::set_unique(&af_indices, true);
    let mut block_ids = vec![0u64; af_block_ids.elements()];
    af_block_ids.host(&mut block_ids);
    (block_ids, af_indices, af_offsets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coord_block() {
        let shape = Shape::from(vec![2, 3, 4]);
        let bounds = coord_bounds(&shape);
        let coords: Vec<Vec<u64>> = Bounds::all(&shape).affected().collect();

        let num_coords = coords.len() as u64;
        let (block_ids, af_indices, af_offsets) = coord_block(
            coords.into_iter(),
            &bounds,
            PER_BLOCK,
            shape.len(),
            num_coords,
        );

        let mut indices = vec![0u64; af_indices.elements()];
        af_indices.host(&mut indices);

        let mut offsets = vec![0u64; af_offsets.elements()];
        af_offsets.host(&mut offsets);

        assert_eq!(block_ids, vec![0]);
        assert_eq!(indices, vec![0; 24]);
        assert_eq!(offsets, (0..24).collect::<Vec<u64>>());
    }

    #[test]
    fn test_block_offsets() {
        let shape = Shape::from(vec![2, 3, 4]);
        let bounds = coord_bounds(&shape);
        let coords: Vec<Vec<u64>> = Bounds::all(&shape).affected().collect();

        let num_coords = coords.len() as u64;
        let (block_ids, af_indices, af_offsets) = coord_block(
            coords.into_iter(),
            &bounds,
            PER_BLOCK,
            shape.len(),
            num_coords,
        );

        let (af_block_offsets, new_start) = block_offsets(&af_indices, &af_offsets, 0f64, 0u64);
        let mut block_offsets = vec![0u64; af_block_offsets.elements()];
        af_block_offsets.host(&mut block_offsets);

        assert_eq!(new_start, 24f64);
        assert_eq!(block_offsets, (0..24).collect::<Vec<u64>>());
    }
}
