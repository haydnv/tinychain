use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::iter;
use std::ops;
use std::sync::Arc;

use async_trait::async_trait;
use num::Integer;

use crate::error;
use crate::transaction::{Txn, TxnId};
use crate::value::{TCResult, Value};

#[derive(Clone)]
pub enum AxisIndex {
    At(u64),
    In(ops::Range<u64>, u64),
    Of(Vec<u64>),
}

impl AxisIndex {
    fn all(dim: u64) -> AxisIndex {
        AxisIndex::In(0..dim, 1)
    }
}

impl From<u64> for AxisIndex {
    fn from(at: u64) -> AxisIndex {
        AxisIndex::At(at)
    }
}

impl From<Vec<u64>> for AxisIndex {
    fn from(of: Vec<u64>) -> AxisIndex {
        AxisIndex::Of(of)
    }
}

impl From<(ops::Range<u64>, u64)> for AxisIndex {
    fn from(slice: (ops::Range<u64>, u64)) -> AxisIndex {
        AxisIndex::In(slice.0, slice.1)
    }
}

impl fmt::Display for AxisIndex {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use AxisIndex::*;
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

#[derive(Clone)]
pub struct Index {
    axes: Vec<AxisIndex>,
}

impl Index {
    pub fn all(shape: &Shape) -> Index {
        shape
            .0
            .iter()
            .map(|dim| AxisIndex::In(0..*dim, 1))
            .collect::<Vec<AxisIndex>>()
            .into()
    }

    pub fn to_coord(self) -> Vec<u64> {
        let mut indices = Vec::with_capacity(self.len());
        for i in self.axes {
            match i {
                AxisIndex::At(i) => indices.push(i),
                _ => panic!("Expected u64 but found {}", i),
            }
        }
        indices
    }

    pub fn len(&self) -> usize {
        self.axes.len()
    }

    pub fn normalize(&mut self, shape: &Shape) {
        assert!(self.len() <= shape.len());

        for axis in self.axes.len()..shape.len() {
            self.axes.push(AxisIndex::all(shape[axis]))
        }
    }
}

impl<Idx: std::slice::SliceIndex<[AxisIndex]>> ops::Index<Idx> for Index {
    type Output = Idx::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.axes[index]
    }
}

impl<Idx: std::slice::SliceIndex<[AxisIndex]>> ops::IndexMut<Idx> for Index {
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output {
        &mut self.axes[index]
    }
}

impl From<Vec<AxisIndex>> for Index {
    fn from(axes: Vec<AxisIndex>) -> Index {
        Index { axes }
    }
}

impl From<&[u64]> for Index {
    fn from(coord: &[u64]) -> Index {
        let axes = coord.iter().map(|i| AxisIndex::At(*i)).collect();
        Index { axes }
    }
}

impl fmt::Display for Index {
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

#[derive(Clone)]
pub struct Shape(Vec<u64>);

impl Shape {
    pub fn contains(&self, coord: &Index) -> bool {
        if coord.len() > self.len() {
            return false;
        }

        for axis in 0..coord.len() {
            let size = &self[axis];
            match &coord[axis] {
                AxisIndex::At(i) => {
                    if i > size {
                        return false;
                    }
                }
                AxisIndex::In(range, _) => {
                    if range.start > *size || range.end > *size {
                        return false;
                    }
                }
                AxisIndex::Of(indices) => {
                    for i in indices {
                        if i > size {
                            return false;
                        }
                    }
                }
            }
        }

        true
    }

    pub fn selection_shape(&self, coord: &Index) -> Shape {
        assert!(self.contains(coord));

        let mut shape = Vec::with_capacity(self.len());
        for axis in 0..coord.len() {
            match &coord[axis] {
                AxisIndex::At(_) => {}
                AxisIndex::In(range, step) => {
                    let dim = (range.end - range.start).div_ceil(&step);
                    shape.push(dim)
                }
                AxisIndex::Of(indices) => shape.push(indices.len() as u64),
            }
        }
        shape.into()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn size(&self) -> u64 {
        self.0.iter().product()
    }

    pub fn to_vec(&self) -> Vec<u64> {
        self.0.to_vec()
    }
}

impl<Idx: std::slice::SliceIndex<[u64]>> ops::Index<Idx> for Shape {
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
pub trait TensorView: Sized + Send + Sync {
    fn ndim(&self) -> usize;

    fn shape(&'_ self) -> &'_ Shape;

    fn size(&self) -> u64;

    async fn all(&self, txn_id: &TxnId) -> TCResult<bool>;

    async fn any(&self, txn_id: &TxnId) -> TCResult<bool>;

    async fn at(&self, txn_id: &TxnId, coord: &[u64]) -> TCResult<Value>;
}

#[async_trait]
pub trait Broadcast: TensorView {
    type Broadcast: TensorView;

    async fn broadcast(&self, txn: &Arc<Txn>, shape: Shape) -> TCResult<Self::Broadcast>;
}

#[async_trait]
pub trait Expand: TensorView {
    type Expansion: TensorView;

    async fn broadcast(&self, txn: &Arc<Txn>, shape: Shape) -> TCResult<Self::Expansion>;
}

#[async_trait]
pub trait Slice: TensorView {
    type Slice: TensorView;

    async fn slice(&self, txn: &Arc<Txn>, coord: Index) -> TCResult<Self::Slice>;
}

pub trait Rebase: TensorView {
    type Source: TensorView + Slice;

    fn invert_coord(&self, coord: Index) -> Index;

    fn map_coord(&self, source_coord: Index) -> Index;

    fn source(&'_ self) -> &'_ Self::Source;
}

#[async_trait]
pub trait Transpose: TensorView {
    type Permutation: TensorView;

    async fn transpose(
        &self,
        txn: &Arc<Txn>,
        permutation: Option<Vec<usize>>,
    ) -> TCResult<Self::Permutation>;
}

pub struct TensorBroadcast<T: TensorView> {
    source: T,
    shape: Shape,
    size: u64,
    ndim: usize,
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
        })
    }
}

#[async_trait]
impl<T: TensorView + Slice> TensorView for TensorBroadcast<T> {
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
            .at(txn_id, &self.invert_coord(coord.into()).to_coord())
            .await
    }
}

impl<T: TensorView + Slice> Rebase for TensorBroadcast<T> {
    type Source = T;

    fn invert_coord(&self, coord: Index) -> Index {
        let source_ndim = self.source.ndim();
        let mut source_coord = Vec::with_capacity(source_ndim);
        for axis in 0..source_ndim {
            if self.broadcast[axis + self.offset] {
                source_coord.push(AxisIndex::from(0))
            } else {
                source_coord.push(coord[axis + self.offset].clone())
            }
        }

        source_coord.into()
    }

    fn map_coord(&self, source_coord: Index) -> Index {
        let mut coord = Index::all(&self.shape);

        for axis in 0..self.ndim {
            if !self.broadcast[axis + self.offset] {
                coord[axis + self.offset] = source_coord[axis].clone();
            }
        }

        coord
    }

    fn source(&'_ self) -> &'_ Self::Source {
        &self.source
    }
}

pub struct Expansion<T: TensorView> {
    source: T,
    shape: Shape,
    size: u64,
    ndim: usize,
    expand: usize,
}

impl<T: TensorView> Expansion<T> {
    fn new(source: T, expand: usize) -> Expansion<T> {
        assert!(expand < source.ndim());

        let mut shape = source.shape().0.to_vec();
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
            .at(txn_id, &self.invert_coord(coord.into()).to_coord())
            .await
    }
}

impl<T: TensorView + Slice> Rebase for Expansion<T> {
    type Source = T;

    fn invert_coord(&self, mut coord: Index) -> Index {
        if coord.len() >= self.expand {
            coord.axes.remove(self.expand);
        }

        coord
    }

    fn map_coord(&self, mut coord: Index) -> Index {
        if coord.len() >= self.expand {
            coord.axes.insert(self.expand, 0.into());
        }

        coord
    }

    fn source(&'_ self) -> &'_ Self::Source {
        &self.source
    }
}

pub struct Permutation<T: TensorView + Slice> {
    source: T,
    shape: Shape,
    size: u64,
    ndim: usize,
    permutation: Vec<usize>,
}

impl<T: TensorView + Slice> Permutation<T> {
    fn new(source: T, permutation: Option<Vec<usize>>) -> Permutation<T> {
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
            .at(txn_id, &self.invert_coord(coord.into()).to_coord())
            .await
    }
}

impl<T: TensorView + Slice> Rebase for Permutation<T> {
    type Source = T;

    fn invert_coord(&self, coord: Index) -> Index {
        let mut source_coord = Index::all(self.source.shape());
        for axis in 0..coord.len() {
            source_coord[self.permutation[axis]] = coord[axis].clone();
        }
        source_coord
    }

    fn map_coord(&self, source_coord: Index) -> Index {
        let mut coord = Index::all(&self.shape);
        for axis in 0..source_coord.len() {
            coord[self.permutation[axis]] = source_coord[axis].clone();
        }
        coord
    }

    fn source(&'_ self) -> &'_ Self::Source {
        &self.source
    }
}

#[async_trait]
impl<T: TensorView + Slice> Slice for Permutation<T>
where
    <T as Slice>::Slice: Slice + Transpose,
{
    type Slice = <<T as Slice>::Slice as Transpose>::Permutation;

    async fn slice(&self, txn: &Arc<Txn>, coord: Index) -> TCResult<Self::Slice> {
        let mut permutation: BTreeMap<usize, usize> = self
            .permutation()
            .to_vec()
            .into_iter()
            .enumerate()
            .collect();

        let mut elided = HashSet::new();
        for axis in 0..coord.len() {
            if let AxisIndex::At(_) = coord[axis] {
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
        self.source()
            .slice(txn, self.invert_coord(coord))
            .await?
            .transpose(txn, Some(permutation))
            .await
    }
}

pub struct TensorSlice<T: TensorView> {
    source: T,
    shape: Shape,
    size: u64,
    ndim: usize,
    slice: Index,
    offset: HashMap<usize, u64>,
    elided: HashSet<usize>,
}

impl<T: TensorView> TensorSlice<T> {
    fn new(source: T, slice: Index) -> TCResult<TensorSlice<T>> {
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

#[async_trait]
impl<T: TensorView + Slice> TensorView for TensorSlice<T> {
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
            .at(txn_id, &self.invert_coord(coord.into()).to_coord())
            .await
    }
}

impl<T: TensorView + Slice> Rebase for TensorSlice<T> {
    type Source = T;

    fn invert_coord(&self, mut coord: Index) -> Index {
        coord.normalize(&self.shape);

        let mut source_coord = Vec::with_capacity(self.source.ndim());
        let mut source_axis = 0;
        for axis in 0..self.ndim {
            if self.elided.contains(&axis) {
                source_coord.push(self.slice[axis].clone());
                continue;
            }

            use AxisIndex::*;
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

    fn map_coord(&self, source_coord: Index) -> Index {
        assert!(source_coord.len() == self.ndim);

        let mut coord = Vec::with_capacity(self.ndim);

        for axis in 0..self.source.ndim() {
            if self.elided.contains(&axis) {
                continue;
            }

            use AxisIndex::*;
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
                coord.push(AxisIndex::all(self.shape[axis]))
            }
        }

        coord.into()
    }

    fn source(&'_ self) -> &'_ Self::Source {
        &self.source
    }
}
