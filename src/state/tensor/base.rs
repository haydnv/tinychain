use std::collections::{BTreeMap, HashMap, HashSet};
use std::iter;
use std::sync::Arc;

use async_trait::async_trait;
use num::Integer;

use crate::error;
use crate::transaction::{Txn, TxnId};
use crate::value::{TCResult, TCType, Value};

use super::dense::BlockTensor;
use super::index::*;

#[async_trait]
pub trait TensorView: Sized + Send + Sync {
    fn dtype(&self) -> TCType;

    fn ndim(&self) -> usize;

    fn shape(&'_ self) -> &'_ Shape;

    fn size(&self) -> u64;

    async fn all(&self, txn_id: &TxnId) -> TCResult<bool>;

    async fn any(&self, txn_id: &TxnId) -> TCResult<bool>;

    async fn at<'a>(&'a self, txn_id: &'a TxnId, coord: Vec<u64>) -> TCResult<Value>;
}

#[async_trait]
pub trait ToDense: TensorView {
    async fn to_dense(self: Arc<Self>, txn: Arc<Txn>) -> TCResult<BlockTensor>;
}

pub trait Broadcast: TensorView {
    fn broadcast(self: Arc<Self>, shape: Shape) -> TCResult<Arc<TensorBroadcast<Self>>>;
}

pub trait Expand: TensorView {
    type Expansion: TensorView;

    fn expand_dims(self: Arc<Self>, axis: usize) -> TCResult<Self::Expansion>;
}

pub trait Slice: TensorView {
    type Slice: TensorView;

    fn slice(self: Arc<Self>, coord: Index) -> TCResult<Arc<Self::Slice>>;
}

pub trait Rebase: TensorView {
    type Source: TensorView + Slice;

    fn invert_index(&self, index: Index) -> Index;

    fn map_index(&self, source_index: Index) -> Index;

    fn source(&self) -> Arc<Self::Source>;
}

pub trait Transpose: TensorView {
    type Permutation: TensorView;

    fn transpose(
        self: Arc<Self>,
        permutation: Option<Vec<usize>>,
    ) -> TCResult<Arc<Self::Permutation>>;
}

pub struct TensorBroadcast<T: TensorView> {
    source: Arc<T>,
    shape: Shape,
    size: u64,
    ndim: usize,
    broadcast: Vec<bool>,
    offset: usize,
}

impl<T: TensorView> TensorBroadcast<T> {
    fn new(source: Arc<T>, shape: Shape) -> TCResult<Arc<TensorBroadcast<T>>> {
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

        Ok(Arc::new(TensorBroadcast {
            source,
            shape,
            size,
            ndim,
            broadcast,
            offset,
        }))
    }
}

#[async_trait]
impl<T: TensorView + Slice> TensorView for TensorBroadcast<T> {
    fn dtype(&self) -> TCType {
        self.source.dtype()
    }

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

    async fn at<'a>(&'a self, txn_id: &'a TxnId, coord: Vec<u64>) -> TCResult<Value> {
        self.source
            .at(txn_id, self.invert_index(coord.into()).to_coord())
            .await
    }
}

impl<T: TensorView + Slice> Rebase for TensorBroadcast<T> {
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

    fn source(&self) -> Arc<Self::Source> {
        self.source.clone()
    }
}

pub struct Expansion<T: TensorView> {
    source: Arc<T>,
    shape: Shape,
    size: u64,
    ndim: usize,
    expand: usize,
}

impl<T: TensorView> Expansion<T> {
    fn new(source: Arc<T>, expand: usize) -> Expansion<T> {
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
        }
    }
}

#[async_trait]
impl<T: TensorView + Slice> TensorView for Expansion<T> {
    fn dtype(&self) -> TCType {
        self.source().dtype()
    }

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

    async fn at<'a>(&'a self, txn_id: &'a TxnId, coord: Vec<u64>) -> TCResult<Value> {
        self.source
            .at(txn_id, self.invert_index(coord.into()).to_coord())
            .await
    }
}

impl<T: TensorView + Slice> Rebase for Expansion<T> {
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

    fn source(&self) -> Arc<Self::Source> {
        self.source.clone()
    }
}

pub struct Permutation<T: TensorView + Slice> {
    source: Arc<T>,
    shape: Shape,
    size: u64,
    ndim: usize,
    permutation: Vec<usize>,
}

impl<T: TensorView + Slice> Permutation<T> {
    fn new(source: Arc<T>, permutation: Option<Vec<usize>>) -> Permutation<T> {
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
        }
    }

    pub fn permutation(&'_ self) -> &'_ [usize] {
        &self.permutation
    }
}

#[async_trait]
impl<T: TensorView + Slice> TensorView for Permutation<T> {
    fn dtype(&self) -> TCType {
        self.source().dtype()
    }

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

    async fn at<'a>(&'a self, txn_id: &'a TxnId, coord: Vec<u64>) -> TCResult<Value> {
        self.source
            .at(txn_id, self.invert_index(coord.into()).to_coord())
            .await
    }
}

impl<T: TensorView + Slice> Rebase for Permutation<T> {
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

    fn source(&self) -> Arc<Self::Source> {
        self.source.clone()
    }
}

impl<T: TensorView + Slice> Slice for Permutation<T>
where
    <T as Slice>::Slice: Slice + Transpose,
{
    type Slice = <<T as Slice>::Slice as Transpose>::Permutation;

    fn slice(self: Arc<Self>, index: Index) -> TCResult<Arc<Self::Slice>> {
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
        slice.transpose(Some(permutation))
    }
}

pub struct TensorSlice<T: TensorView> {
    source: Arc<T>,
    shape: Shape,
    size: u64,
    ndim: usize,
    slice: Index,
    offset: HashMap<usize, u64>,
    elided: HashSet<usize>,
}

impl<T: TensorView> TensorSlice<T> {
    pub fn new(source: Arc<T>, slice: Index) -> TCResult<TensorSlice<T>> {
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
        })
    }
}

impl<T: TensorView + Slice> Slice for TensorSlice<T> {
    type Slice = TensorSlice<T>;

    fn slice(self: Arc<Self>, index: Index) -> TCResult<Arc<Self::Slice>> {
        Ok(Arc::new(TensorSlice::new(
            self.source.clone(),
            self.invert_index(index),
        )?))
    }
}

#[async_trait]
impl<T: TensorView + Slice> TensorView for TensorSlice<T> {
    fn dtype(&self) -> TCType {
        self.source().dtype()
    }

    fn ndim(&self) -> usize {
        self.ndim
    }

    fn shape(&'_ self) -> &'_ Shape {
        &self.shape
    }

    fn size(&self) -> u64 {
        self.size
    }

    async fn all(&self, _txn_id: &TxnId) -> TCResult<bool> {
        panic!("NOT IMPLEMENTED")
    }

    async fn any(&self, _txn_id: &TxnId) -> TCResult<bool> {
        panic!("NOT IMPLEMENTED")
    }

    async fn at<'a>(&'a self, txn_id: &'a TxnId, coord: Vec<u64>) -> TCResult<Value> {
        self.source
            .at(txn_id, self.invert_index(coord.into()).to_coord())
            .await
    }
}

impl<T: TensorView + Slice> Rebase for TensorSlice<T> {
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

    fn source(&self) -> Arc<Self::Source> {
        self.source.clone()
    }
}
