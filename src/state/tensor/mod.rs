use std::fmt;
use std::iter;
use std::ops::{Bound, Index};
use std::slice::SliceIndex;

use arrayfire::HasAfEnum;
use async_trait::async_trait;

use crate::error;
use crate::transaction::TxnId;
use crate::value::TCResult;

mod dense;
mod sparse;

#[derive(Clone)]
pub enum AxisSlice {
    At(u64),
    In(Bound<i64>, Bound<i64>),
    Stride(Bound<i64>, Bound<i64>, u64),
}

impl AxisSlice {
    fn unbounded() -> AxisSlice {
        AxisSlice::In(Bound::Unbounded, Bound::Unbounded)
    }
}

impl From<u64> for AxisSlice {
    fn from(at: u64) -> AxisSlice {
        AxisSlice::At(at)
    }
}

impl fmt::Display for AxisSlice {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use AxisSlice::*;
        match self {
            At(at) => write!(f, "{}", at),
            In(start, stop) => write!(f, "{}", format_bound(start, stop)),
            Stride(start, stop, step) => write!(f, "{} step {}", format_bound(start, stop), step),
        }
    }
}

pub struct Slice {
    axes: Vec<AxisSlice>,
}

impl Slice {
    fn len(&self) -> usize {
        self.axes.len()
    }
}

impl<Idx: SliceIndex<[AxisSlice]>> Index<Idx> for Slice {
    type Output = Idx::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.axes[index]
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

impl<Idx: SliceIndex<[u64]>> Index<Idx> for Shape {
    type Output = Idx::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.0[index]
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
pub trait TensorView: Send + Sync {
    type DType: HasAfEnum;

    fn ndim(&self) -> usize;

    fn shape(&'_ self) -> &'_ Shape;

    fn size(&self) -> u64;

    async fn all(&self, txn_id: &TxnId) -> TCResult<bool>;

    async fn any(&self, txn_id: &TxnId) -> TCResult<bool>;

    async fn slice(&self, txn_id: &TxnId, slice: Slice) -> TCResult<TensorSlice>;
}

pub enum Tensor {
    Dense(dense::BlockTensor),
    Sparse(sparse::SparseTensor),
}

trait Rebase: TensorView {
    fn invert_coord(&self, coord: Slice) -> TCResult<Slice>;

    fn map_coord(&self, source_coord: Slice) -> TCResult<Slice>;
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

    async fn slice(&self, txn_id: &TxnId, coord: Slice) -> TCResult<TensorSlice> {
        self.source.slice(txn_id, self.invert_coord(coord)?).await
    }
}

impl<T: TensorView> Rebase for Broadcast<T> {
    fn invert_coord(&self, coord: Slice) -> TCResult<Slice> {
        if coord.len() > self.ndim {
            return Err(error::bad_request("Invalid coordinate", coord));
        }

        let source_ndim = self.source.ndim();
        let mut source_coord = Vec::with_capacity(source_ndim);
        for axis in 0..source_ndim {
            if self.broadcast[axis + self.offset] {
                source_coord.push(AxisSlice::from(0))
            } else {
                source_coord.push(coord[axis + self.offset].clone())
            }
        }

        Ok(source_coord.into())
    }

    fn map_coord(&self, source_coord: Slice) -> TCResult<Slice> {
        let mut coord: Vec<AxisSlice> = iter::repeat(AxisSlice::unbounded())
            .take(self.ndim)
            .collect();
        for axis in 0..self.ndim {
            if !self.broadcast[axis + self.offset] {
                coord[axis + self.offset] = source_coord[axis].clone();
            }
        }

        Ok(coord.into())
    }
}

pub struct TensorSlice {}

fn format_bound(start: &Bound<i64>, stop: &Bound<i64>) -> String {
    use Bound::*;

    if start == &Unbounded && stop == &Unbounded {
        return "[...]".to_string();
    }

    let start = match start {
        Included(i) => format!("[{}", i),
        Excluded(i) => format!("({}", i),
        Unbounded => "[...".to_string(),
    };

    let stop = match stop {
        Included(i) => format!("{}]", i),
        Excluded(i) => format!("{})", i),
        Unbounded => "...]".to_string(),
    };

    format!("{}{}", start, stop)
}
