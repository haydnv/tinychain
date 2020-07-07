use std::collections::{HashMap, HashSet};
use std::fmt;
use std::iter;
use std::ops::{Index, IndexMut, Range};

use arrayfire::HasAfEnum;
use async_trait::async_trait;
use num::Integer;

use crate::error;
use crate::transaction::TxnId;
use crate::value::TCResult;

mod dense;
mod sparse;

pub enum Tensor {
    Dense(dense::BlockTensor),
    Sparse(sparse::SparseTensor),
}

#[derive(Clone)]
enum AxisSlice {
    At(u64),
    In(Range<u64>, u64),
    Of(Vec<u64>),
}

impl AxisSlice {
    fn all(dim: u64) -> AxisSlice {
        AxisSlice::In(0..dim, 1)
    }
}

impl From<u64> for AxisSlice {
    fn from(at: u64) -> AxisSlice {
        AxisSlice::At(at)
    }
}

impl From<Vec<u64>> for AxisSlice {
    fn from(of: Vec<u64>) -> AxisSlice {
        AxisSlice::Of(of)
    }
}

impl From<(Range<u64>, u64)> for AxisSlice {
    fn from(slice: (Range<u64>, u64)) -> AxisSlice {
        AxisSlice::In(slice.0, slice.1)
    }
}

impl fmt::Display for AxisSlice {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use AxisSlice::*;
        match self {
            At(at) => write!(f, "{}", at),
            In(range, 1) => write!(f, "[{}, {})", range.start, range.end),
            In(range, step) => write!(f, "[{}, {}) step {}", range.start, range.end, step),
            Of(indices) => write!(
                f,
                "({})",
                indices
                    .iter()
                    .map(|i| format!("{}", i))
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
        }
    }
}

struct Slice {
    axes: Vec<AxisSlice>,
}

impl Slice {
    fn all(shape: &Shape) -> Slice {
        shape
            .0
            .iter()
            .map(|dim| AxisSlice::In(0..*dim, 1))
            .collect::<Vec<AxisSlice>>()
            .into()
    }

    fn len(&self) -> usize {
        self.axes.len()
    }

    fn normalize(&mut self, shape: &Shape) {
        assert!(self.len() <= shape.len());

        for axis in self.axes.len()..shape.len() {
            self.axes.push(AxisSlice::all(shape[axis]))
        }
    }
}

impl<Idx: std::slice::SliceIndex<[AxisSlice]>> Index<Idx> for Slice {
    type Output = Idx::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.axes[index]
    }
}

impl<Idx: std::slice::SliceIndex<[AxisSlice]>> IndexMut<Idx> for Slice {
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output {
        &mut self.axes[index]
    }
}

impl From<Vec<AxisSlice>> for Slice {
    fn from(axes: Vec<AxisSlice>) -> Slice {
        Slice { axes }
    }
}

impl fmt::Display for Slice {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "[{}]",
            self.axes
                .iter()
                .map(|axis| format!("{}", axis))
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}

pub struct Shape(Vec<u64>);

impl Shape {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn size(&self) -> u64 {
        self.0.iter().product()
    }
}

impl<Idx: std::slice::SliceIndex<[u64]>> Index<Idx> for Shape {
    type Output = Idx::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.0[index]
    }
}

impl From<Vec<u64>> for Shape {
    fn from(shape: Vec<u64>) -> Shape {
        Shape(shape)
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "[{}]",
            self.0
                .iter()
                .map(|dim| format!("{}", dim))
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}

#[async_trait]
trait TensorView: Send + Sync {
    type DType: HasAfEnum;
    type SliceType: TensorView;

    fn ndim(&self) -> usize;

    fn shape(&'_ self) -> &'_ Shape;

    fn size(&self) -> u64;

    async fn all(&self, txn_id: &TxnId) -> TCResult<bool>;

    async fn any(&self, txn_id: &TxnId) -> TCResult<bool>;

    async fn slice(&self, txn_id: &TxnId, slice: Slice) -> TCResult<TensorSlice<Self::SliceType>>;
}

trait Rebase: TensorView {
    fn invert_coord(&self, coord: Slice) -> Slice;

    fn map_coord(&self, source_coord: Slice) -> Slice;
}

struct Broadcast<T: TensorView> {
    source: T,
    shape: Shape,
    size: u64,
    ndim: usize,
    broadcast: Vec<bool>,
    offset: usize,
}

impl<T: TensorView> Broadcast<T> {
    fn new(source: T, shape: Shape) -> TCResult<Broadcast<T>> {
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

        Ok(Broadcast {
            source,
            shape,
            size,
            ndim,
            broadcast,
            offset,
        })
    }
}

#[async_trait]
impl<T: TensorView> TensorView for Broadcast<T> {
    type DType = T::DType;
    type SliceType = T::SliceType;

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

    async fn slice(&self, txn_id: &TxnId, coord: Slice) -> TCResult<TensorSlice<T::SliceType>> {
        self.source.slice(txn_id, self.invert_coord(coord)).await
    }
}

impl<T: TensorView> Rebase for Broadcast<T> {
    fn invert_coord(&self, coord: Slice) -> Slice {
        let source_ndim = self.source.ndim();
        let mut source_coord = Vec::with_capacity(source_ndim);
        for axis in 0..source_ndim {
            if self.broadcast[axis + self.offset] {
                source_coord.push(AxisSlice::from(0))
            } else {
                source_coord.push(coord[axis + self.offset].clone())
            }
        }

        source_coord.into()
    }

    fn map_coord(&self, source_coord: Slice) -> Slice {
        let mut coord = Slice::all(&self.shape);

        for axis in 0..self.ndim {
            if !self.broadcast[axis + self.offset] {
                coord[axis + self.offset] = source_coord[axis].clone();
            }
        }

        coord
    }
}

struct Permutation<T: TensorView> {
    source: T,
    shape: Shape,
    size: u64,
    ndim: usize,
    permute_from: HashMap<usize, usize>,
    permute_to: HashMap<usize, usize>,
}

impl<T: TensorView> Permutation<T> {
    fn new(source: T, permutation: Option<Vec<usize>>) -> Permutation<T> {
        let ndim = source.ndim();
        let mut permutation = permutation
            .or_else(|| {
                let mut axes: Vec<usize> = (0..ndim).collect();
                axes.reverse();
                Some(axes)
            })
            .unwrap();

        assert!(permutation.len() == ndim);

        let source_shape = source.shape();
        let mut permute_from: HashMap<usize, usize> = HashMap::new();
        let mut permute_to: HashMap<usize, usize> = HashMap::new();
        let mut shape: Vec<u64> = Vec::with_capacity(ndim);
        for (source_axis, dest_axis) in permutation.iter().enumerate() {
            permute_from.insert(source_axis, *dest_axis);
            permute_to.insert(*dest_axis, source_axis);
            shape.push(source_shape[*dest_axis]);
        }

        let shape: Shape = shape.into();
        let size = shape.size();
        Permutation {
            source,
            shape,
            size,
            ndim,
            permute_from,
            permute_to,
        }
    }
}

#[async_trait]
impl<T: TensorView> TensorView for Permutation<T> {
    type DType = T::DType;
    type SliceType = T::SliceType;

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

    async fn slice(&self, txn_id: &TxnId, coord: Slice) -> TCResult<TensorSlice<T::SliceType>> {
        self.source.slice(txn_id, self.invert_coord(coord)).await
    }
}

impl<T: TensorView> Rebase for Permutation<T> {
    fn invert_coord(&self, coord: Slice) -> Slice {
        let mut source_coord = Slice::all(self.source.shape());
        for axis in 0..coord.len() {
            source_coord[self.permute_to[&axis]] = coord[axis].clone();
        }
        source_coord
    }

    fn map_coord(&self, source_coord: Slice) -> Slice {
        let mut coord = Slice::all(&self.shape);
        for axis in 0..source_coord.len() {
            coord[self.permute_from[&axis]] = source_coord[axis].clone();
        }
        coord
    }
}

struct TensorSlice<T: TensorView> {
    source: T,
    shape: Shape,
    size: u64,
    ndim: usize,
    slice: Slice,
    offset: HashMap<usize, u64>,
    elided: HashSet<usize>,
}

impl<T: TensorView> TensorSlice<T> {
    fn new(source: T, slice: Slice) -> TCResult<TensorSlice<T>> {
        let mut shape: Vec<u64> = Vec::with_capacity(slice.len());
        let mut offset = HashMap::new();
        let mut elided = HashSet::new();

        for axis in 0..slice.len() {
            match &slice[axis] {
                AxisSlice::At(_) => {
                    elided.insert(axis);
                }
                AxisSlice::In(range, step) => {
                    let dim = (range.end - range.start).div_ceil(step);
                    shape.push(dim);
                    offset.insert(axis, range.start);
                }
                AxisSlice::Of(indices) => shape.push(indices.len() as u64),
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

#[async_trait]
impl<T: TensorView> TensorView for TensorSlice<T> {
    type DType = T::DType;
    type SliceType = T::SliceType;

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

    async fn slice(&self, txn_id: &TxnId, coord: Slice) -> TCResult<TensorSlice<T::SliceType>> {
        self.source.slice(txn_id, self.invert_coord(coord)).await
    }
}

impl<T: TensorView> Rebase for TensorSlice<T> {
    fn invert_coord(&self, mut coord: Slice) -> Slice {
        coord.normalize(&self.shape);

        let mut source_coord = Vec::with_capacity(self.source.ndim());
        let mut source_axis = 0;
        for axis in 0..self.ndim {
            if self.elided.contains(&axis) {
                source_coord.push(self.slice[axis].clone());
                continue;
            }

            use AxisSlice::*;
            match &coord[source_axis] {
                In(range, this_step) => {
                    if let In(source_range, source_step) = &self.slice[axis] {
                        let start = range.start + source_range.start;
                        let end = start + (source_step * (range.end - range.start));
                        let step = source_step * this_step;
                        source_coord.push((start..end, step).into());
                    } else {
                        assert!(range.start == 0);
                        source_coord.push(self.slice[axis].clone());
                    }
                }
                Of(indices) => {
                    let offset = self.offset.get(&axis).unwrap_or(&0);
                    source_coord.push(
                        indices
                            .iter()
                            .map(|i| i + offset)
                            .collect::<Vec<u64>>()
                            .into(),
                    )
                }
                At(i) => {
                    let offset = self.offset.get(&axis).unwrap_or(&0);
                    source_coord.push((i + offset).into())
                }
            }
            source_axis += 1;
        }

        source_coord.into()
    }

    fn map_coord(&self, source_coord: Slice) -> Slice {
        assert!(source_coord.len() == self.ndim);

        let mut coord = Vec::with_capacity(self.ndim);

        for axis in 0..self.source.ndim() {
            if self.elided.contains(&axis) {
                continue;
            }

            use AxisSlice::*;
            match &source_coord[axis] {
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

        for axis in source_coord.len()..self.ndim {
            if !self.elided.contains(&axis) {
                coord.push(AxisSlice::all(self.shape[axis]))
            }
        }

        coord.into()
    }
}

fn index_error(axis: usize, index: i64, dim: u64) -> error::TCError {
    error::bad_request(
        &format!(
            "Index is out of bounds for axis {} with dimension {}",
            axis, dim
        ),
        index,
    )
}
