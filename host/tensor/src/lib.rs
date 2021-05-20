use std::fmt;

use number_general::NumberType;

use tcgeneric::{path_label, Class, Instance, NativeClass, PathLabel, PathSegment, TCPathBuf};

pub use bounds::Shape;

#[allow(dead_code)]
mod bounds;
mod dense;

type Coord = Vec<u64>;

const PREFIX: PathLabel = path_label(&["state", "collection", "tensor"]);

pub trait TensorAccess: Send {
    fn dtype(&self) -> NumberType;

    fn ndim(&self) -> usize;

    fn shape(&'_ self) -> &'_ Shape;

    fn size(&self) -> u64;
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum TensorType {}

impl Class for TensorType {}

impl NativeClass for TensorType {
    fn from_path(path: &[PathSegment]) -> Option<Self> {
        if path.len() > 3 && &path[..3] == &PREFIX[..] {
            todo!()
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
pub enum Tensor {}

impl Instance for Tensor {
    type Class = TensorType;

    fn class(&self) -> Self::Class {
        todo!()
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("a Tensor")
    }
}
