use std::iter;

use crate::error;
use crate::value::TCResult;

use super::bounds::{AxisBounds, Bounds, Shape};

#[derive(Clone)]
pub struct Broadcast {
    source_shape: Shape,
    shape: Shape,
    broadcast: Vec<bool>,
    offset: usize,
    inverted_axes: Vec<usize>,
}

impl Broadcast {
    pub fn new(source_shape: Shape, shape: Shape) -> TCResult<Broadcast> {
        let ndim = shape.len();
        if source_shape.len() > shape.len() {
            return Err(error::bad_request(
                &format!("Cannot broadcast into {}", shape),
                source_shape,
            ));
        }

        let offset = ndim - source_shape.len();
        let mut inverted_axes = Vec::with_capacity(shape.len());
        let mut broadcast: Vec<bool> = iter::repeat(true).take(ndim).collect();

        for axis in offset..ndim {
            if shape[axis] == source_shape[axis - offset] {
                broadcast[axis] = false;
                inverted_axes.push(axis);
            } else if shape[axis] == 1 || source_shape[axis - offset] == 1 {
                inverted_axes.push(axis - offset);
            } else {
                return Err(error::bad_request(
                    &format!("Cannot broadcast into {}", shape),
                    source_shape,
                ));
            }
        }

        Ok(Broadcast {
            source_shape,
            shape,
            broadcast,
            offset,
            inverted_axes,
        })
    }

    pub fn invert_axes(&self, axes: Vec<usize>) -> Vec<usize> {
        let mut inverted: Vec<usize> = axes.iter().map(|x| self.inverted_axes[*x]).collect();
        inverted.dedup();
        inverted
    }

    pub fn invert_bounds(&self, bounds: Bounds) -> Bounds {
        let source_ndim = self.source_shape.len();
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

    pub fn invert_coord(&self, coord: Vec<u64>) -> Vec<u64> {
        let source_ndim = self.source_shape.len();
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

    pub fn map_bounds(&self, source_bounds: Bounds) -> Bounds {
        assert!(source_bounds.len() == self.source_shape.len());

        let mut bounds = Bounds::all(&self.shape);

        for axis in 0..self.source_shape.len() {
            if !self.broadcast[axis + self.offset] {
                bounds[axis + self.offset] = source_bounds[axis].clone();
            }
        }

        bounds
    }

    pub fn shape(&'_ self) -> &'_ Shape {
        &self.shape
    }
}

pub struct Transpose {
    source_shape: Shape,
    shape: Shape,
    permutation: Vec<usize>,
}

impl Transpose {
    pub fn new(source_shape: Shape, permutation: Option<Vec<usize>>) -> TCResult<Transpose> {
        let ndim = source_shape.len();
        let permutation = permutation
            .or_else(|| {
                let mut axes: Vec<usize> = (0..ndim).collect();
                axes.reverse();
                Some(axes)
            })
            .unwrap();

        if permutation.len() != ndim {
            let permutation: Vec<String> = permutation.iter().map(|x| x.to_string()).collect();
            return Err(error::bad_request(
                "Invalid permutation for transpose",
                format!(
                    "Tensor with shape {} cannot transpose axes ({})",
                    source_shape,
                    permutation.join(", ")
                ),
            ));
        }

        let mut shape: Vec<u64> = Vec::with_capacity(ndim);
        for axis in &permutation {
            shape.push(source_shape[*axis]);
        }
        let shape: Shape = shape.into();
        Ok(Transpose {
            source_shape,
            shape,
            permutation,
        })
    }

    pub fn invert_axes(&self, axes: Vec<usize>) -> Vec<usize> {
        axes.iter().map(|x| self.permutation[*x]).collect()
    }

    pub fn invert_bounds(&self, bounds: Bounds) -> Bounds {
        let mut source_bounds = Bounds::all(&self.source_shape);
        for axis in 0..bounds.len() {
            source_bounds[self.permutation[axis]] = bounds[axis].clone();
        }
        source_bounds
    }

    pub fn invert_coord(&self, coord: Vec<u64>) -> Vec<u64> {
        let mut source_coord = Vec::with_capacity(self.source_shape.len());
        for axis in 0..coord.len() {
            source_coord[self.permutation[axis]] = coord[axis];
        }

        source_coord
    }

    pub fn map_bounds(&self, source_bounds: Bounds) -> Bounds {
        let mut bounds = Bounds::all(&self.shape);
        for axis in 0..source_bounds.len() {
            bounds[self.permutation[axis]] = source_bounds[axis].clone();
        }
        bounds
    }

    pub fn map_coord(&self, source_coord: Vec<u64>) -> Vec<u64> {
        let mut coord = Vec::with_capacity(self.source_shape.len());
        for axis in 0..source_coord.len() {
            coord[self.permutation[axis]] = source_coord[axis];
        }

        coord
    }

    pub fn shape(&'_ self) -> &'_ Shape {
        &self.shape
    }
}
