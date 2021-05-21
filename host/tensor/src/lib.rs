use std::fmt;
use std::pin::Pin;

use futures::Future;
use number_general::{Number, NumberType};

use tc_error::*;
use tc_transact::fs::Dir;
use tc_transact::Transaction;
use tcgeneric::{path_label, Class, Instance, NativeClass, PathLabel, PathSegment, TCPathBuf};

pub use bounds::{Bounds, Shape};
pub use dense::DenseTensor;

#[allow(dead_code)]
mod bounds;
mod dense;

const PREFIX: PathLabel = path_label(&["state", "collection", "tensor"]);

type Coord = Vec<u64>;

type Read<'a> = Pin<Box<dyn Future<Output = TCResult<(Coord, Number)>> + Send + 'a>>;

pub trait ReadValueAt<D: Dir, T: Transaction<D>> {
    fn read_value_at<'a>(&'a self, txn: &'a T, coord: Coord) -> Read<'a>;
}

pub trait TensorAccess: Send {
    fn dtype(&self) -> NumberType;

    fn ndim(&self) -> usize;

    fn shape(&'_ self) -> &'_ Shape;

    fn size(&self) -> u64;
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum TensorType {
    Dense,
}

impl Class for TensorType {}

impl NativeClass for TensorType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if path.len() == 4 && &path[..3] == &PREFIX[..] {
            match path[3].as_str() {
                "dense" => Some(Self::Dense),
                "sparse" => todo!(),
                _ => None,
            }
        } else {
            None
        }
    }

    fn path(&self) -> TCPathBuf {
        PREFIX.into()
    }
}

impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("type Tensor")
    }
}

#[derive(Clone)]
pub enum Tensor<F, D, T> {
    Dense(DenseTensor<F, dense::BlockListFile<F, D, T>>),
}

impl<F, D, T> Instance for Tensor<F, D, T>
where
    Self: Send + Sync,
{
    type Class = TensorType;

    fn class(&self) -> Self::Class {
        match self {
            Self::Dense(_) => TensorType::Dense,
        }
    }
}

impl<F, D, T> fmt::Display for Tensor<F, D, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a Tensor")
    }
}
