use std::collections::{BTreeMap, HashMap, HashSet};
use std::iter;
use std::sync::Arc;

use async_trait::async_trait;
use itertools::Itertools;
use num::Integer;

use crate::error;
use crate::transaction::{Txn, TxnId};
use crate::value::class::NumberType;
use crate::value::{Number, TCResult};

use super::bounds::*;
use super::dense::{BlockTensor, DenseTensorView};
use super::sparse::{SparseTensorView, TableTensor};
use super::TensorView;

#[async_trait]
pub trait DenseTensorUnary {
    async fn as_dtype(self, txn: Arc<Txn>, dtype: NumberType) -> TCResult<BlockTensor>;

    async fn copy(self, txn: Arc<Txn>) -> TCResult<BlockTensor>;

    async fn abs(self, txn: Arc<Txn>) -> TCResult<BlockTensor>;

    async fn sum(self, txn: Arc<Txn>, axis: usize) -> TCResult<BlockTensor>;

    async fn sum_all(self, txn_id: TxnId) -> TCResult<Number>;

    async fn product(self, txn: Arc<Txn>, axis: usize) -> TCResult<BlockTensor>;

    async fn product_all(self, txn_id: TxnId) -> TCResult<Number>;

    async fn not(self, txn: Arc<Txn>) -> TCResult<BlockTensor>;
}

#[async_trait]
pub trait SparseTensorUnary: SparseTensorView {
    async fn as_dtype(self, txn: Arc<Txn>, dtype: NumberType) -> TCResult<TableTensor>;

    async fn copy(self, txn: Arc<Txn>) -> TCResult<TableTensor>;

    async fn abs(self, txn: Arc<Txn>) -> TCResult<TableTensor>;

    async fn sum(self, txn: Arc<Txn>, axis: usize) -> TCResult<TableTensor>;

    async fn sum_all(self, txn_id: TxnId) -> TCResult<Number>;

    async fn product(self, txn: Arc<Txn>, axis: usize) -> TCResult<TableTensor>;

    async fn product_all(self, txn_id: TxnId) -> TCResult<Number>;

    async fn not(self, txn: Arc<Txn>) -> TCResult<TableTensor>;
}

#[async_trait]
pub trait DenseTensorArithmetic<Object: DenseTensorView> {
    async fn add(self, other: Object, txn: Arc<Txn>) -> TCResult<BlockTensor>;

    async fn multiply(self, other: Object, txn: Arc<Txn>) -> TCResult<BlockTensor>;
}

#[async_trait]
pub trait SparseTensorArithmetic<Object: SparseTensorView> {
    async fn add(self, other: Object, txn: Arc<Txn>) -> TCResult<TableTensor>;

    async fn multiply(self, other: Object, txn: Arc<Txn>) -> TCResult<TableTensor>;
}

#[async_trait]
pub trait DenseTensorBoolean<Object: DenseTensorView> {
    async fn and(self, other: Object, txn: Arc<Txn>) -> TCResult<BlockTensor>;

    async fn or(self, other: Object, txn: Arc<Txn>) -> TCResult<BlockTensor>;

    async fn xor(self, other: Object, txn: Arc<Txn>) -> TCResult<BlockTensor>;
}

#[async_trait]
pub trait SparseTensorBoolean<Object: SparseTensorView> {
    async fn and(self, other: Object, txn: Arc<Txn>) -> TCResult<TableTensor>;

    async fn or(self, other: Object, txn: Arc<Txn>) -> TCResult<TableTensor>;

    async fn xor(self, other: Object, txn: Arc<Txn>) -> TCResult<TableTensor>;
}

#[async_trait]
pub trait DenseTensorCompare<Object: DenseTensorView> {
    async fn equals(self, other: Object, txn: Arc<Txn>) -> TCResult<BlockTensor>;

    async fn gt(self, other: Object, txn: Arc<Txn>) -> TCResult<BlockTensor>;

    async fn gte(self, other: Object, txn: Arc<Txn>) -> TCResult<BlockTensor>;

    async fn lt(self, other: Object, txn: Arc<Txn>) -> TCResult<BlockTensor>;

    async fn lte(self, other: Object, txn: Arc<Txn>) -> TCResult<BlockTensor>;
}

#[async_trait]
pub trait SparseTensorCompare<Object: SparseTensorView> {
    async fn equals(self, other: Object, txn: Arc<Txn>) -> TCResult<BlockTensor>;

    async fn gt(self, other: Object, txn: Arc<Txn>) -> TCResult<TableTensor>;

    async fn gte(self, other: Object, txn: Arc<Txn>) -> TCResult<BlockTensor>;

    async fn lt(self, other: Object, txn: Arc<Txn>) -> TCResult<TableTensor>;

    async fn lte(self, other: Object, txn: Arc<Txn>) -> TCResult<BlockTensor>;
}

#[async_trait]
pub trait AnyAll: TensorView {
    async fn all(self, txn_id: TxnId) -> TCResult<bool>;

    async fn any(self, txn_id: TxnId) -> TCResult<bool>;
}

pub trait Broadcast: TensorView {
    fn broadcast(self, shape: Shape) -> TCResult<TensorBroadcast<Self>>;
}

pub trait Expand: TensorView {
    type Expansion: TensorView;

    fn expand_dims(self, axis: usize) -> TCResult<Self::Expansion>;
}

pub trait Slice: TensorView {
    type Slice: TensorView;

    fn slice(self, coord: Bounds) -> TCResult<Self::Slice>;
}

pub trait Rebase: TensorView {
    type Source: TensorView;

    fn invert_bounds(&self, bounds: Bounds) -> Bounds;

    fn invert_coord(&self, coord: Vec<u64>) -> Vec<u64>;

    fn map_bounds(&self, source_bounds: Bounds) -> Bounds;

    fn map_coord(&self, source_coord: Vec<u64>) -> Vec<u64>;

    fn source(&'_ self) -> &'_ Self::Source;
}

pub trait Transpose: TensorView {
    type Permutation: TensorView;

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Permutation>;
}

#[derive(Clone)]
pub struct TensorBroadcast<T: TensorView> {
    source: T,
    shape: Shape,
    broadcast: Vec<bool>,
    offset: usize,
}

impl<T: TensorView> TensorBroadcast<T> {
    fn new(source: T, shape: Shape) -> TCResult<TensorBroadcast<T>> {
        let ndim = shape.len();
        if source.ndim() > ndim {
            return Err(error::bad_request(
                &format!("Cannot broadcast into {}", shape),
                source.shape(),
            ));
        }

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
            broadcast,
            offset,
        })
    }
}

#[async_trait]
impl<T: TensorView> TensorView for TensorBroadcast<T> {
    fn dtype(&self) -> NumberType {
        self.source.dtype()
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
impl<T: TensorView + AnyAll> AnyAll for TensorBroadcast<T> {
    async fn all(self, txn_id: TxnId) -> TCResult<bool> {
        self.source.clone().all(txn_id).await
    }

    async fn any(self, txn_id: TxnId) -> TCResult<bool> {
        self.source.clone().any(txn_id).await
    }
}

impl<T: TensorView> Rebase for TensorBroadcast<T> {
    type Source = T;

    fn invert_bounds(&self, bounds: Bounds) -> Bounds {
        let source_ndim = self.source.ndim();
        let mut source_bounds = Vec::with_capacity(source_ndim);
        for axis in 0..source_ndim {
            if self.broadcast[axis + self.offset] {
                source_bounds.push(AxisBounds::from(0))
            } else {
                source_bounds.push(bounds[axis + self.offset].clone())
            }
        }

        source_bounds.into()
    }

    fn invert_coord(&self, coord: Vec<u64>) -> Vec<u64> {
        let source_ndim = self.source.ndim();
        let mut source_coord = Vec::with_capacity(source_ndim);
        for axis in 0..source_ndim {
            if self.broadcast[axis + self.offset] {
                source_coord.push(0);
            } else {
                source_coord.push(coord[axis + self.offset]);
            }
        }

        source_coord
    }

    fn map_bounds(&self, source_bounds: Bounds) -> Bounds {
        let mut bounds = Bounds::all(&self.shape);

        for axis in 0..self.ndim() {
            if !self.broadcast[axis + self.offset] {
                bounds[axis + self.offset] = source_bounds[axis].clone();
            }
        }

        bounds
    }

    fn map_coord(&self, source_coord: Vec<u64>) -> Vec<u64> {
        let mut coord = Vec::with_capacity(self.ndim());

        for axis in 0..self.ndim() {
            if !self.broadcast[axis + self.offset] {
                coord[axis + self.offset] = source_coord[axis];
            }
        }

        coord
    }

    fn source(&'_ self) -> &'_ Self::Source {
        &self.source
    }
}

#[derive(Clone)]
pub struct Expansion<T: TensorView> {
    source: T,
    shape: Shape,
    expand: HashMap<usize, usize>,
}

impl<T: TensorView> Expansion<T> {
    fn new(source: T, expand: HashMap<usize, usize>) -> TCResult<Expansion<T>> {
        if expand.len() > source.ndim() {
            return Err(error::bad_request(
                "Invalid expansion requested for tensor with shape",
                source.shape(),
            ));
        }

        for (axis, dims) in expand.iter() {
            if axis > &source.ndim() {
                return Err(error::bad_request("Dimension out of range", axis));
            } else if *dims == 0 {
                return Err(error::bad_request("Invalid dimension expansion", dims));
            }
        }

        let mut shape = vec![];
        for source_axis in 0..source.ndim() {
            if let Some(dims) = expand.get(&source_axis) {
                shape.extend(iter::repeat(1).take(*dims));
            }
            shape.push(source.shape()[source_axis]);
        }
        let shape: Shape = shape.into();

        Ok(Expansion {
            source,
            shape,
            expand,
        })
    }
}

impl<T: TensorView> TensorView for Expansion<T> {
    fn dtype(&self) -> NumberType {
        self.source().dtype()
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
impl<T: TensorView + AnyAll> AnyAll for Expansion<T> {
    async fn all(self, txn_id: TxnId) -> TCResult<bool> {
        self.source.clone().all(txn_id).await
    }

    async fn any(self, txn_id: TxnId) -> TCResult<bool> {
        self.source.clone().any(txn_id).await
    }
}

impl<T: TensorView> Rebase for Expansion<T> {
    type Source = T;

    fn invert_bounds(&self, bounds: Bounds) -> Bounds {
        let mut bounds = bounds.to_vec();

        if bounds.len() == self.ndim() && self.expand.contains_key(&self.source().ndim()) {
            bounds.pop();
        }

        for axis in 0..Ord::min(self.source().ndim(), bounds.len()) {
            if let Some(dims) = self.expand.get(&axis) {
                bounds.drain(axis..Ord::min(bounds.len(), axis + dims));
            }
        }

        bounds.into()
    }

    fn invert_coord(&self, mut coord: Vec<u64>) -> Vec<u64> {
        if coord.len() == self.ndim() && self.expand.contains_key(&self.source().ndim()) {
            coord.pop();
        }

        for axis in 0..Ord::min(self.source().ndim(), coord.len()) {
            if let Some(dims) = self.expand.get(&axis) {
                coord.drain(axis..Ord::min(coord.len(), axis + dims));
            }
        }

        coord
    }

    fn map_bounds(&self, source_bounds: Bounds) -> Bounds {
        assert!(source_bounds.len() == self.source.ndim());

        let mut bounds = Vec::with_capacity(self.ndim());

        for axis in 0..self.source.ndim() {
            if let Some(dims) = self.expand.get(&axis) {
                bounds.extend(iter::repeat(AxisBounds::At(0)).take(*dims));
            }
            bounds.push(source_bounds[axis].clone());
        }

        if let Some(dims) = self.expand.get(&self.source.ndim()) {
            bounds.extend(iter::repeat(AxisBounds::At(0)).take(*dims));
        }

        bounds.into()
    }

    fn map_coord(&self, source_coord: Vec<u64>) -> Vec<u64> {
        assert!(source_coord.len() == self.source.ndim());

        let mut coord = Vec::with_capacity(self.ndim());

        for (axis, x) in source_coord.iter().enumerate() {
            if let Some(dims) = self.expand.get(&axis) {
                coord.extend(iter::repeat(0).take(*dims));
            }
            coord.push(*x);
        }

        if let Some(dims) = self.expand.get(&self.source.ndim()) {
            coord.extend(iter::repeat(0).take(*dims));
        }

        coord
    }

    fn source(&'_ self) -> &'_ Self::Source {
        &self.source
    }
}

impl<T: TensorView> Expand for Expansion<T> {
    type Expansion = Self;

    fn expand_dims(self, axis: usize) -> TCResult<Self::Expansion> {
        let mut offset = 0;
        let mut expand = self.expand.clone();
        for source_axis in 0..self.source().ndim() {
            if let Some(dims) = expand.get_mut(&source_axis) {
                let start = source_axis + offset;
                offset += *dims;
                let stop = source_axis + offset;
                if (start..stop).contains(&axis) {
                    *dims += 1;
                }
            } else if axis == source_axis + offset {
                expand.insert(source_axis, 1);
            }
        }

        if axis == self.ndim() {
            if let Some(dims) = expand.get_mut(&self.source().ndim()) {
                *dims += 1;
            } else {
                expand.insert(self.source().ndim(), 1);
            }
        }

        Expansion::new(self.source().clone(), expand)
    }
}

#[derive(Clone)]
pub struct Permutation<T: TensorView> {
    source: T,
    shape: Shape,
    permutation: Vec<usize>,
}

impl<T: TensorView> Permutation<T> {
    pub fn new(source: T, permutation: Option<Vec<usize>>) -> TCResult<Permutation<T>> {
        let ndim = source.ndim();
        let permutation = permutation
            .or_else(|| {
                let mut axes: Vec<usize> = (0..ndim).collect();
                axes.reverse();
                Some(axes)
            })
            .unwrap();

        if permutation.len() != ndim {
            return Err(error::bad_request(
                "Invalid permutation for transpose",
                format!(
                    "Tensor with shape {} cannot transpose axes ({})",
                    source.shape(),
                    permutation.iter().map(|x| x.to_string()).join(", ")
                ),
            ));
        }

        let source_shape = source.shape();
        let mut shape: Vec<u64> = Vec::with_capacity(ndim);
        for axis in &permutation {
            shape.push(source_shape[*axis]);
        }
        let shape: Shape = shape.into();
        Ok(Permutation {
            source,
            shape,
            permutation,
        })
    }

    pub fn permutation(&'_ self) -> &'_ [usize] {
        &self.permutation
    }
}

impl<T: TensorView> TensorView for Permutation<T> {
    fn dtype(&self) -> NumberType {
        self.source().dtype()
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
impl<T: TensorView + Transpose + AnyAll> AnyAll for Permutation<T> {
    async fn all(self, txn_id: TxnId) -> TCResult<bool> {
        self.source.clone().all(txn_id).await
    }

    async fn any(self, txn_id: TxnId) -> TCResult<bool> {
        self.source.clone().any(txn_id).await
    }
}

impl<T: TensorView> Rebase for Permutation<T> {
    type Source = T;

    fn invert_bounds(&self, bounds: Bounds) -> Bounds {
        let mut source_bounds = Bounds::all(self.source.shape());
        for axis in 0..bounds.len() {
            source_bounds[self.permutation[axis]] = bounds[axis].clone();
        }
        source_bounds
    }

    fn invert_coord(&self, coord: Vec<u64>) -> Vec<u64> {
        let mut source_coord = Vec::with_capacity(self.source.ndim());
        for axis in 0..coord.len() {
            source_coord[self.permutation[axis]] = coord[axis];
        }

        source_coord
    }

    fn map_bounds(&self, source_bounds: Bounds) -> Bounds {
        let mut bounds = Bounds::all(&self.shape);
        for axis in 0..source_bounds.len() {
            bounds[self.permutation[axis]] = source_bounds[axis].clone();
        }
        bounds
    }

    fn map_coord(&self, source_coord: Vec<u64>) -> Vec<u64> {
        let mut coord = Vec::with_capacity(self.source.ndim());
        for axis in 0..source_coord.len() {
            coord[self.permutation[axis]] = source_coord[axis];
        }

        coord
    }

    fn source(&'_ self) -> &'_ Self::Source {
        &self.source
    }
}

impl<T: TensorView + Slice> Slice for Permutation<T>
where
    <T as Slice>::Slice: Transpose,
{
    type Slice = <<T as Slice>::Slice as Transpose>::Permutation;

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        let mut permutation: BTreeMap<usize, usize> = self
            .permutation()
            .to_vec()
            .into_iter()
            .enumerate()
            .collect();

        let mut elided = HashSet::new();
        for axis in 0..bounds.len() {
            if let AxisBounds::At(_) = bounds[axis] {
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
        let source_bounds = self.invert_bounds(bounds);
        let source = self.source().clone();
        let slice = source.slice(source_bounds)?;
        slice.transpose(Some(permutation))
    }
}

impl<T: TensorView> Transpose for Permutation<T> {
    type Permutation = Self;

    fn transpose(self, permutation: Option<Vec<usize>>) -> TCResult<Self::Permutation> {
        let permutation = permutation.unwrap_or((0..self.ndim()).rev().collect());

        // map this axis -> source axis
        let this_permutation: HashMap<usize, usize> =
            self.permutation().iter().cloned().enumerate().collect();
        // map requested axis -> source axis
        let permutation = permutation
            .iter()
            .map(|requested_axis| this_permutation.get(requested_axis).unwrap())
            .cloned()
            .collect();

        Permutation::new(self.source().clone(), Some(permutation))
    }
}

#[derive(Clone)]
pub struct TensorSlice<T: TensorView> {
    source: T,
    shape: Shape,
    bounds: Bounds,
    offset: HashMap<usize, u64>,
    elided: HashMap<usize, u64>,
}

impl<T: TensorView> TensorSlice<T> {
    pub fn new(source: T, bounds: Bounds) -> TCResult<TensorSlice<T>> {
        let mut shape: Vec<u64> = Vec::with_capacity(bounds.len());
        let mut offset = HashMap::new();
        let mut elided = HashMap::new();

        for axis in 0..bounds.len() {
            match &bounds[axis] {
                AxisBounds::At(c) => {
                    elided.insert(axis, *c);
                }
                AxisBounds::In(range, step) => {
                    let dim = (range.end - range.start).div_ceil(step);
                    shape.push(dim);
                    offset.insert(axis, range.start);
                }
                AxisBounds::Of(indices) => shape.push(indices.len() as u64),
            }
        }

        let shape: Shape = shape.into();

        Ok(TensorSlice {
            source,
            shape,
            bounds,
            offset,
            elided,
        })
    }

    pub fn bounds(&'_ self) -> &'_ Bounds {
        &self.bounds
    }
}

#[async_trait]
impl<T: TensorView> TensorView for TensorSlice<T> {
    fn dtype(&self) -> NumberType {
        Rebase::source(self).dtype()
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

impl<T: TensorView> Slice for TensorSlice<T> {
    type Slice = TensorSlice<T>;

    fn slice(self, bounds: Bounds) -> TCResult<Self::Slice> {
        TensorSlice::new(self.source.clone(), self.invert_bounds(bounds))
    }
}

impl<T: TensorView> Rebase for TensorSlice<T> {
    type Source = T;

    fn invert_bounds(&self, mut bounds: Bounds) -> Bounds {
        bounds.normalize(&self.shape);

        let mut source_bounds = Vec::with_capacity(self.source.ndim());
        let mut source_axis = 0;
        for axis in 0..self.ndim() {
            if let Some(c) = self.elided.get(&axis) {
                source_bounds.push(AxisBounds::At(*c));
                continue;
            }

            use AxisBounds::*;
            match &bounds[source_axis] {
                In(range, this_step) => {
                    if let In(source_range, source_step) = &self.bounds[axis] {
                        let start = range.start + source_range.start;
                        let end = start + (source_step * (range.end - range.start));
                        let step = source_step * this_step;
                        source_bounds.push((start..end, step).into());
                    } else {
                        assert!(range.start == 0);
                        source_bounds.push(self.bounds[axis].clone());
                    }
                }
                Of(indices) => {
                    let offset = self.offset.get(&axis).unwrap_or(&0);
                    source_bounds.push(
                        indices
                            .iter()
                            .map(|i| i + offset)
                            .collect::<Vec<u64>>()
                            .into(),
                    )
                }
                At(i) => {
                    let offset = self.offset.get(&axis).unwrap_or(&0);
                    source_bounds.push((i + offset).into())
                }
            }
            source_axis += 1;
        }

        source_bounds.into()
    }

    fn invert_coord(&self, coord: Vec<u64>) -> Vec<u64> {
        let mut source_coord = Vec::with_capacity(self.source.ndim());
        let mut source_axis = 0;
        for axis in 0..self.ndim() {
            if let Some(elided) = self.elided.get(&axis) {
                source_coord.push(*elided);
            } else {
                let offset = self.offset.get(&axis).unwrap_or(&0);
                source_coord.push(coord[source_axis] + *offset);
                source_axis += 1;
            }
        }

        source_coord
    }

    fn map_bounds(&self, source_bounds: Bounds) -> Bounds {
        assert!(source_bounds.len() == self.source().ndim());

        let mut coord: Vec<AxisBounds> = Vec::with_capacity(self.ndim());

        for axis in 0..self.source.ndim() {
            if self.elided.contains_key(&axis) {
                continue;
            }

            use AxisBounds::*;
            match &source_bounds[axis] {
                In(_, _) => todo!(),
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

        coord.into()
    }

    fn map_coord(&self, source_coord: Vec<u64>) -> Vec<u64> {
        assert!(source_coord.len() == self.source().ndim());
        let mut coord = Vec::with_capacity(self.ndim());
        for (axis, c) in source_coord.iter().enumerate() {
            if self.elided.contains_key(&axis) {
                continue;
            }

            let offset = self.offset.get(&axis).unwrap_or(&0);
            coord.push(c - offset);
        }

        coord
    }

    fn source(&'_ self) -> &'_ Self::Source {
        &self.source
    }
}
