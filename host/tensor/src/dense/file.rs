use std::iter::FromIterator;
use std::marker::PhantomData;

use afarray::Array;
use async_trait::async_trait;
use futures::future;
use futures::stream::{self, StreamExt, TryStreamExt};
use log::debug;
use num::integer::div_ceil;
use number_general::{Number, NumberInstance, NumberType};

use tc_error::*;
use tc_transact::fs::{BlockData, BlockId, Dir, File};
use tc_transact::{Transaction, TxnId};
use tc_value::Value;
use tcgeneric::{TCBoxTryFuture, TCTryStream};

use crate::{Bounds, Coord, Read, ReadValueAt, Shape, TensorAccess};

use super::{block_offsets, coord_block, DenseAccess, DenseAccessor};

#[derive(Clone)]
pub struct BlockListFile<F, D, T> {
    file: F,
    dtype: NumberType,
    shape: Shape,
    dir: PhantomData<D>,
    txn: PhantomData<T>,
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

    fn block_stream<'a>(&'a self, txn: &'a T) -> TCBoxTryFuture<'a, TCTryStream<'a, Array>> {
        Box::pin(async move {
            let file = &self.file;
            let block_stream = Box::pin(
                stream::iter(0..(div_ceil(self.size(), Array::max_size())))
                    .map(BlockId::from)
                    .then(move |block_id| file.read_block(*txn.id(), block_id))
                    .map_ok(|block| (*block).clone()),
            );

            let block_stream: TCTryStream<'a, Array> = Box::pin(block_stream);
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

fn coord_bounds(shape: &Shape) -> Coord {
    (0..shape.len())
        .map(|axis| shape[axis + 1..].iter().product())
        .collect()
}
