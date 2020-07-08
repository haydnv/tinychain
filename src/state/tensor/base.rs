use std::collections::{BTreeMap, HashMap, HashSet};
use std::iter;
use std::marker::PhantomData;

use async_trait::async_trait;
use num::Integer;

use crate::error;
use crate::transaction::TxnId;
use crate::value::{TCResult, Value};

use super::index::*;

#[async_trait]
pub trait TensorView<'a>: Sized + Send + Sync {
    fn ndim(&self) -> usize;

    fn shape(&'_ self) -> &'_ Shape;

    fn size(&self) -> u64;

    async fn all(&self, txn_id: &TxnId) -> TCResult<bool>;

    async fn any(&self, txn_id: &TxnId) -> TCResult<bool>;

    async fn at(&self, txn_id: &TxnId, coord: &[u64]) -> TCResult<Value>;
}

#[async_trait]
pub trait Broadcast<'a>: TensorView<'a> {
    type Broadcast: TensorView<'a>;

    async fn broadcast(&'a self, shape: Shape) -> TCResult<Self::Broadcast>;
}

#[async_trait]
pub trait Expand<'a>: TensorView<'a> {
    type Expansion: TensorView<'a>;

    async fn expand_dims(&'a self, axis: usize) -> TCResult<Self::Expansion>;
}

pub trait Slice<'a>: TensorView<'a> {
    type Slice: TensorView<'a>;

    fn slice(&'a self, coord: Index) -> TCResult<Self::Slice>;
}

pub trait Rebase<'a>: TensorView<'a> {
    type Source: TensorView<'a> + Slice<'a>;

    fn invert_index(&self, index: Index) -> Index;

    fn map_index(&self, source_index: Index) -> Index;

    fn source(&'a self) -> &'a Self::Source;
}

pub trait Transpose<'a>: TensorView<'a> {
    type Permutation: TensorView<'a>;

    fn transpose(&'a self, permutation: Option<Vec<usize>>) -> TCResult<Self::Permutation>;

    fn transpose_into(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Permutation>;
}

pub struct TensorBroadcast<'a, T: TensorView<'a>> {
    source: T,
    shape: Shape,
    size: u64,
    ndim: usize,
    broadcast: Vec<bool>,
    offset: usize,
    phantom: PhantomData<&'a T>,
}

impl<'a, T: TensorView<'a>> TensorBroadcast<'a, T> {
    fn new(source: T, shape: Shape) -> TCResult<TensorBroadcast<'a, T>> {
        let ndim = shape.len();
        if source.ndim() > ndim {
            return Err(error::bad_request(
                &format!("Cannot broadcast into {}", shape),
                source.shape(),
            ));
        }

        let size = shape.size();
        let offset = ndim - source.ndim();
        let mut broadcast: Vec<bool> = iter::repeat(true).take(ndim).collect();

        let source_shape = source.shape();
        for axis in offset..ndim {
            if shape[axis] == source_shape[axis - offset] {
                broadcast[axis] = false;
            } else if shape[axis] == 1 || source_shape[axis - offset] == 1 {
                // no-op
            } else {
                return Err(error::bad_request(
                    &format!("Cannot broadcast into {}", shape),
                    source_shape,
                ));
            }
        }

        Ok(TensorBroadcast {
            source,
            shape,
            size,
            ndim,
            broadcast,
            offset,
            phantom: PhantomData,
        })
    }
}

#[async_trait]
impl<'a, T: TensorView<'a> + Slice<'a>> TensorView<'a> for TensorBroadcast<'a, T> {
    fn ndim(&self) -> usize {
        self.ndim
    }

    fn shape(&'_ self) -> &'_ Shape {
        &self.shape
    }

    fn size(&self) -> u64 {
        self.size
    }

    async fn all(&self, txn_id: &TxnId) -> TCResult<bool> {
        self.source.all(txn_id).await
    }

    async fn any(&self, txn_id: &TxnId) -> TCResult<bool> {
        self.source.any(txn_id).await
    }

    async fn at(&self, txn_id: &TxnId, coord: &[u64]) -> TCResult<Value> {
        self.source
            .at(txn_id, &self.invert_index(coord.into()).to_coord())
            .await
    }
}

impl<'a, T: TensorView<'a> + Slice<'a>> Rebase<'a> for TensorBroadcast<'a, T> {
    type Source = T;

    fn invert_index(&self, index: Index) -> Index {
        let source_ndim = self.source.ndim();
        let mut source_index = Vec::with_capacity(source_ndim);
        for axis in 0..source_ndim {
            if self.broadcast[axis + self.offset] {
                source_index.push(AxisIndex::from(0))
            } else {
                source_index.push(index[axis + self.offset].clone())
            }
        }

        source_index.into()
    }

    fn map_index(&self, source_index: Index) -> Index {
        let mut index = Index::all(&self.shape);

        for axis in 0..self.ndim {
            if !self.broadcast[axis + self.offset] {
                index[axis + self.offset] = source_index[axis].clone();
            }
        }

        index
    }

    fn source(&'a self) -> &'a Self::Source {
        &self.source
    }
}

pub struct Expansion<'a, T: TensorView<'a>> {
    source: T,
    shape: Shape,
    size: u64,
    ndim: usize,
    expand: usize,
    phantom: PhantomData<&'a T>,
}

impl<'a, T: TensorView<'a>> Expansion<'a, T> {
    fn new(source: T, expand: usize) -> Expansion<'a, T> {
        assert!(expand < source.ndim());

        let mut shape = source.shape().to_vec();
        shape.insert(expand, 1);

        let shape: Shape = shape.into();
        let size = shape.size();
        let ndim = shape.len();
        Expansion {
            source,
            shape,
            size,
            ndim,
            expand,
            phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<'a, T: TensorView<'a> + Slice<'a>> TensorView<'a> for Expansion<'a, T> {
    fn ndim(&self) -> usize {
        self.ndim
    }

    fn shape(&'_ self) -> &'_ Shape {
        &self.shape
    }

    fn size(&self) -> u64 {
        self.size
    }

    async fn all(&self, txn_id: &TxnId) -> TCResult<bool> {
        self.source.all(txn_id).await
    }

    async fn any(&self, txn_id: &TxnId) -> TCResult<bool> {
        self.source.any(txn_id).await
    }

    async fn at(&self, txn_id: &TxnId, coord: &[u64]) -> TCResult<Value> {
        self.source
            .at(txn_id, &self.invert_index(coord.into()).to_coord())
            .await
    }
}

impl<'a, T: TensorView<'a> + Slice<'a>> Rebase<'a> for Expansion<'a, T> {
    type Source = T;

    fn invert_index(&self, mut index: Index) -> Index {
        if index.len() >= self.expand {
            index.axes.remove(self.expand);
        }

        index
    }

    fn map_index(&self, mut index: Index) -> Index {
        if index.len() >= self.expand {
            index.axes.insert(self.expand, 0.into());
        }

        index
    }

    fn source(&'a self) -> &'a Self::Source {
        &self.source
    }
}

pub struct Permutation<'a, T: TensorView<'a> + Slice<'a>> {
    source: T,
    shape: Shape,
    size: u64,
    ndim: usize,
    permutation: Vec<usize>,
    phantom: PhantomData<&'a T>,
}

impl<'a, T: TensorView<'a> + Slice<'a>> Permutation<'a, T> {
    fn new(source: T, permutation: Option<Vec<usize>>) -> Permutation<'a, T> {
        let ndim = source.ndim();
        let permutation = permutation
            .or_else(|| {
                let mut axes: Vec<usize> = (0..ndim).collect();
                axes.reverse();
                Some(axes)
            })
            .unwrap();

        assert!(permutation.len() == ndim);

        let source_shape = source.shape();
        let mut shape: Vec<u64> = Vec::with_capacity(ndim);
        for axis in &permutation {
            shape.push(source_shape[*axis]);
        }
        let shape: Shape = shape.into();
        let size = shape.size();
        Permutation {
            source,
            shape,
            size,
            ndim,
            permutation,
            phantom: PhantomData,
        }
    }

    pub fn permutation(&'a self) -> &'a [usize] {
        &self.permutation
    }
}

#[async_trait]
impl<'a, T: TensorView<'a> + Slice<'a>> TensorView<'a> for Permutation<'a, T> {
    fn ndim(&self) -> usize {
        self.ndim
    }

    fn shape(&'_ self) -> &'_ Shape {
        &self.shape
    }

    fn size(&self) -> u64 {
        self.size
    }

    async fn all(&self, txn_id: &TxnId) -> TCResult<bool> {
        self.source.all(txn_id).await
    }

    async fn any(&self, txn_id: &TxnId) -> TCResult<bool> {
        self.source.any(txn_id).await
    }

    async fn at(&self, txn_id: &TxnId, coord: &[u64]) -> TCResult<Value> {
        self.source
            .at(txn_id, &self.invert_index(coord.into()).to_coord())
            .await
    }
}

impl<'a, T: TensorView<'a> + Slice<'a>> Rebase<'a> for Permutation<'a, T> {
    type Source = T;

    fn invert_index(&self, index: Index) -> Index {
        let mut source_index = Index::all(self.source.shape());
        for axis in 0..index.len() {
            source_index[self.permutation[axis]] = index[axis].clone();
        }
        source_index
    }

    fn map_index(&self, source_index: Index) -> Index {
        let mut index = Index::all(&self.shape);
        for axis in 0..source_index.len() {
            index[self.permutation[axis]] = source_index[axis].clone();
        }
        index
    }

    fn source(&'a self) -> &'a Self::Source {
        &self.source
    }
}

impl<'a, T: TensorView<'a> + Slice<'a>> Slice<'a> for Permutation<'a, T>
where
    <T as Slice<'a>>::Slice: Slice<'a> + Transpose<'a>,
{
    type Slice = <<T as Slice<'a>>::Slice as Transpose<'a>>::Permutation;

    fn slice(&'a self, index: Index) -> TCResult<Self::Slice> {
        let mut permutation: BTreeMap<usize, usize> = self
            .permutation()
            .to_vec()
            .into_iter()
            .enumerate()
            .collect();

        let mut elided = HashSet::new();
        for axis in 0..index.len() {
            if let AxisIndex::At(_) = index[axis] {
                elided.insert(axis);
                permutation.remove(&axis);
            }
        }

        for axis in elided {
            permutation = permutation
                .into_iter()
                .map(|(s, d)| if d > axis { (s, d - 1) } else { (s, d) })
                .collect();
        }

        let permutation: Vec<usize> = permutation.values().cloned().collect();
        let source_index = self.invert_index(index);
        let source = self.source();
        let slice = source.slice(source_index)?;
        slice.transpose_into(Some(permutation))
    }
}

pub struct TensorSlice<'a, T: TensorView<'a>> {
    source: &'a T,
    shape: Shape,
    size: u64,
    ndim: usize,
    slice: Index,
    offset: HashMap<usize, u64>,
    elided: HashSet<usize>,
    phantom: PhantomData<&'a T>,
}

impl<'a, T: TensorView<'a>> TensorSlice<'a, T> {
    pub fn new(source: &'a T, slice: Index) -> TCResult<TensorSlice<T>> {
        let mut shape: Vec<u64> = Vec::with_capacity(slice.len());
        let mut offset = HashMap::new();
        let mut elided = HashSet::new();

        for axis in 0..slice.len() {
            match &slice[axis] {
                AxisIndex::At(_) => {
                    elided.insert(axis);
                }
                AxisIndex::In(range, step) => {
                    let dim = (range.end - range.start).div_ceil(step);
                    shape.push(dim);
                    offset.insert(axis, range.start);
                }
                AxisIndex::Of(indices) => shape.push(indices.len() as u64),
            }
        }

        let shape: Shape = shape.into();
        let size = shape.size();
        let ndim = shape.len();

        Ok(TensorSlice {
            source,
            shape,
            size,
            ndim,
            slice,
            offset,
            elided,
            phantom: PhantomData,
        })
    }
}

impl<'a, T: TensorView<'a> + Slice<'a>> Slice<'a> for TensorSlice<'a, T> {
    type Slice = TensorSlice<'a, T>;

    fn slice(&'a self, index: Index) -> TCResult<Self::Slice> {
        TensorSlice::new(self.source, self.invert_index(index))
    }
}

#[async_trait]
impl<'a, T: TensorView<'a> + Slice<'a>> TensorView<'a> for TensorSlice<'a, T> {
    fn ndim(&self) -> usize {
        self.ndim
    }

    fn shape(&'_ self) -> &'_ Shape {
        &self.shape
    }

    fn size(&self) -> u64 {
        self.size
    }

    async fn all(&self, txn_id: &TxnId) -> TCResult<bool> {
        self.source.all(txn_id).await
    }

    async fn any(&self, txn_id: &TxnId) -> TCResult<bool> {
        self.source.any(txn_id).await
    }

    async fn at(&self, txn_id: &TxnId, coord: &[u64]) -> TCResult<Value> {
        self.source
            .at(txn_id, &self.invert_index(coord.into()).to_coord())
            .await
    }
}

impl<'a, T: TensorView<'a> + Slice<'a>> Rebase<'a> for TensorSlice<'a, T> {
    type Source = T;

    fn invert_index(&self, mut index: Index) -> Index {
        index.normalize(&self.shape);

        let mut source_index = Vec::with_capacity(self.source.ndim());
        let mut source_axis = 0;
        for axis in 0..self.ndim {
            if self.elided.contains(&axis) {
                source_index.push(self.slice[axis].clone());
                continue;
            }

            use AxisIndex::*;
            match &index[source_axis] {
                In(range, this_step) => {
                    if let In(source_range, source_step) = &self.slice[axis] {
                        let start = range.start + source_range.start;
                        let end = start + (source_step * (range.end - range.start));
                        let step = source_step * this_step;
                        source_index.push((start..end, step).into());
                    } else {
                        assert!(range.start == 0);
                        source_index.push(self.slice[axis].clone());
                    }
                }
                Of(indices) => {
                    let offset = self.offset.get(&axis).unwrap_or(&0);
                    source_index.push(
                        indices
                            .iter()
                            .map(|i| i + offset)
                            .collect::<Vec<u64>>()
                            .into(),
                    )
                }
                At(i) => {
                    let offset = self.offset.get(&axis).unwrap_or(&0);
                    source_index.push((i + offset).into())
                }
            }
            source_axis += 1;
        }

        source_index.into()
    }

    fn map_index(&self, source_index: Index) -> Index {
        assert!(source_index.len() == self.ndim);

        let mut coord = Vec::with_capacity(self.ndim);

        for axis in 0..self.source.ndim() {
            if self.elided.contains(&axis) {
                continue;
            }

            use AxisIndex::*;
            match &source_index[axis] {
                In(_, _) => panic!("NOT IMPLEMENTED"),
                Of(indices) => {
                    let offset = self.offset.get(&axis).unwrap_or(&0);
                    coord.push(
                        indices
                            .iter()
                            .map(|i| i - offset)
                            .collect::<Vec<u64>>()
                            .into(),
                    );
                }
                At(i) => {
                    let offset = self.offset.get(&axis).unwrap_or(&0);
                    coord.push((i - offset).into())
                }
            }
        }

        for axis in source_index.len()..self.ndim {
            if !self.elided.contains(&axis) {
                coord.push(AxisIndex::all(self.shape[axis]))
            }
        }

        coord.into()
    }

    fn source(&'a self) -> &'a Self::Source {
        &self.source
    }
}
