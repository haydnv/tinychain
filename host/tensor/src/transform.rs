use std::collections::HashMap;
use std::iter;

use tc_error::*;

use crate::bounds::{AxisBounds, Bounds, Shape};

use super::Coord;

pub trait Rebase {
    type Invert;
    type Map;

    fn invert_bounds(&self, bounds: Bounds) -> Bounds;

    fn invert_coord(&self, coord: &[u64]) -> Self::Invert;

    fn map_coord(&self, coord: Coord) -> Self::Map;
}

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
            return Err(TCError::bad_request(
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
                return Err(TCError::bad_request(
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

    pub fn map_bounds(&self, source_bounds: Bounds) -> Bounds {
        assert_eq!(source_bounds.len(), self.source_shape.len());

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

impl Rebase for Broadcast {
    type Invert = Coord;
    type Map = Bounds;

    fn invert_bounds(&self, bounds: Bounds) -> Bounds {
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

    fn invert_coord(&self, coord: &[u64]) -> Self::Invert {
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

    fn map_coord(&self, coord: Coord) -> Self::Map {
        self.map_bounds(coord.into())
    }
}

#[derive(Clone)]
pub struct Expand {
    source_shape: Shape,
    shape: Shape,
    expand: usize,
    inverted_axes: Vec<usize>,
}

impl Expand {
    pub fn new(source_shape: Shape, expand: usize) -> TCResult<Expand> {
        if expand > source_shape.len() {
            return Err(TCError::bad_request("Axis out of bounds", expand));
        }

        let mut inverted_axes = Vec::with_capacity(source_shape.len() + 1);
        inverted_axes.extend(0..source_shape.len());
        inverted_axes.insert(expand, expand);

        let mut shape = source_shape.to_vec();
        shape.insert(expand, 1);
        let shape: Shape = shape.into();

        Ok(Expand {
            source_shape,
            shape,
            expand,
            inverted_axes,
        })
    }

    pub fn expand_axis(&self) -> usize {
        self.expand
    }

    pub fn invert_axes(&self, _axes: Vec<usize>) -> Vec<usize> {
        unimplemented!()
    }

    pub fn shape(&'_ self) -> &'_ Shape {
        &self.shape
    }
}

impl Rebase for Expand {
    type Invert = Coord;
    type Map = Coord;

    fn invert_bounds(&self, mut bounds: Bounds) -> Bounds {
        if bounds.len() < self.expand {
            bounds.remove(self.expand);
        }

        bounds
    }

    fn invert_coord(&self, coord: &[u64]) -> Self::Invert {
        assert_eq!(coord.len(), self.shape.len());

        let mut inverted = Vec::with_capacity(self.source_shape.len());
        inverted.extend(&coord[..self.expand]);

        if self.expand < self.source_shape.len() {
            inverted.extend(&coord[self.expand + 1..]);
        }

        inverted
    }

    fn map_coord(&self, mut coord: Coord) -> Self::Map {
        assert_eq!(coord.len(), self.source_shape.len());
        coord.insert(self.expand, 0);
        coord
    }
}

#[derive(Clone)]
pub struct Reduce {
    source_shape: Shape,
    axis: usize,
    shape: Shape,
}

impl Reduce {
    pub fn new(source_shape: Shape, axis: usize) -> TCResult<Reduce> {
        if axis >= source_shape.len() {
            return Err(TCError::bad_request(
                &format!("Tensor with shape {} has no such axis", source_shape),
                axis,
            ));
        }

        let mut shape = source_shape.clone();
        shape.remove(axis);
        Ok(Reduce {
            source_shape,
            shape,
            axis,
        })
    }

    pub fn axis(&self) -> usize {
        self.axis
    }

    pub fn shape(&'_ self) -> &'_ Shape {
        &self.shape
    }

    pub fn invert_axes(&self, _permutation: Option<Vec<usize>>) -> (Vec<usize>, usize) {
        unimplemented!()
    }

    pub fn reduce_axis(&self, bounds: &Bounds) -> usize {
        let mut reduce_axis = self.axis;
        for bound in bounds.iter() {
            if bound.is_index() {
                reduce_axis -= 1;
            }
        }

        reduce_axis
    }
}

impl Rebase for Reduce {
    type Invert = Bounds;
    type Map = Coord;

    fn invert_bounds(&self, mut bounds: Bounds) -> Bounds {
        if bounds.len() < self.axis {
            bounds
        } else {
            bounds.insert(self.axis, AxisBounds::all(self.source_shape[self.axis]));
            bounds
        }
    }

    fn invert_coord(&self, coord: &[u64]) -> Bounds {
        let mut bounds: Vec<AxisBounds> = coord.iter().map(|i| AxisBounds::At(*i)).collect();
        bounds.insert(self.axis, AxisBounds::all(self.source_shape[self.axis]));
        bounds.into()
    }

    fn map_coord(&self, _coord: Coord) -> Coord {
        panic!("Reduced coordinate has no mapping")
    }
}

#[derive(Clone)]
pub struct Slice {
    source_shape: Shape,
    shape: Shape,
    bounds: Bounds,
    offset: HashMap<usize, u64>,
    elided: HashMap<usize, u64>,
    inverted_axes: Vec<usize>,
}

impl Slice {
    pub fn new(source_shape: Shape, bounds: Bounds) -> TCResult<Slice> {
        source_shape.validate_bounds(&bounds)?;

        let mut shape: Coord = Vec::with_capacity(bounds.len());
        let mut offset = HashMap::new();
        let mut elided = HashMap::new();
        let mut inverted_axes = Vec::with_capacity(bounds.len());

        for axis in 0..bounds.len() {
            match &bounds[axis] {
                AxisBounds::At(c) => {
                    elided.insert(axis, *c);
                }
                AxisBounds::In(range) => {
                    let dim = range.end - range.start;
                    shape.push(dim);
                    offset.insert(axis, range.start);
                    inverted_axes.push(axis);
                }
                AxisBounds::Of(indices) => {
                    shape.push(indices.len() as u64);
                    inverted_axes.push(axis);
                }
            }
        }

        for axis in bounds.len()..source_shape.len() {
            shape.push(source_shape[axis]);
            inverted_axes.push(axis);
        }

        let shape: Shape = shape.into();

        Ok(Slice {
            source_shape,
            shape,
            bounds,
            offset,
            elided,
            inverted_axes,
        })
    }

    pub fn bounds(&'_ self) -> &'_ Bounds {
        &self.bounds
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn shape(&'_ self) -> &'_ Shape {
        &self.shape
    }

    pub fn size(&self) -> u64 {
        self.shape.size()
    }
}

impl Rebase for Slice {
    type Invert = Coord;
    type Map = Coord;

    fn invert_bounds(&self, mut bounds: Bounds) -> Bounds {
        bounds.normalize(&self.shape);

        if bounds.is_empty() || bounds == Bounds::all(self.shape()) {
            return self.bounds.clone();
        }

        let mut source_bounds = Vec::with_capacity(self.source_shape.len());
        let mut source_axis = 0;
        for axis in 0..self.shape.len() {
            if let Some(c) = self.elided.get(&axis) {
                source_bounds.push(AxisBounds::At(*c));
                continue;
            }

            use AxisBounds::*;
            match &bounds[source_axis] {
                In(range) => {
                    if let In(source_range) = &self.bounds[axis] {
                        let start = range.start + source_range.start;
                        let end = start + (range.end - range.start);
                        source_bounds.push((start..end).into());
                    } else {
                        assert_eq!(range.start, 0);
                        source_bounds.push(self.bounds[axis].clone());
                    }
                }
                Of(indices) => {
                    let offset = self.offset.get(&axis).unwrap_or(&0);
                    source_bounds.push(indices.iter().map(|i| i + offset).collect::<Coord>().into())
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

    fn invert_coord(&self, coord: &[u64]) -> Self::Invert {
        assert_eq!(coord.len(), self.shape.len());

        let mut source_coord = Vec::with_capacity(self.source_shape.len());
        let mut source_axis = 0;
        for axis in 0..self.source_shape.len() {
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

    fn map_coord(&self, source_coord: Coord) -> Self::Map {
        assert_eq!(source_coord.len(), self.source_shape.len());

        let mut coord = Vec::with_capacity(self.shape.len());
        for (axis, c) in source_coord.iter().enumerate() {
            if self.elided.contains_key(&axis) {
                continue;
            }

            let offset = self.offset.get(&axis).unwrap_or(&0);
            coord.push(c - offset);
        }

        coord
    }
}

#[derive(Clone)]
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
            return Err(TCError::bad_request(
                "Invalid permutation for transpose",
                format!(
                    "Tensor with shape {} cannot transpose axes ({})",
                    source_shape,
                    permutation.join(", ")
                ),
            ));
        }

        let mut shape: Coord = Vec::with_capacity(ndim);
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

    pub fn invert_axes(&self, axes: &[usize]) -> Vec<usize> {
        axes.iter().map(|x| self.permutation[*x]).collect()
    }

    pub fn invert_permutation(&self, _bounds: &Bounds) -> Vec<usize> {
        todo!()
    }

    pub fn shape(&'_ self) -> &'_ Shape {
        &self.shape
    }
}

impl Rebase for Transpose {
    type Invert = Coord;
    type Map = Coord;

    fn invert_bounds(&self, bounds: Bounds) -> Bounds {
        let mut source_bounds = Bounds::all(&self.source_shape);
        for axis in 0..bounds.len() {
            source_bounds[self.permutation[axis]] = bounds[axis].clone();
        }
        source_bounds
    }

    fn invert_coord(&self, coord: &[u64]) -> Self::Invert {
        assert_eq!(coord.len(), self.permutation.len());

        let mut source_coord = vec![0; coord.len()];
        for axis in 0..coord.len() {
            source_coord[self.permutation[axis]] = coord[axis];
        }

        source_coord
    }

    fn map_coord(&self, source_coord: Coord) -> Self::Map {
        assert_eq!(source_coord.len(), self.permutation.len());

        let mut coord = vec![0; source_coord.len()];
        for axis in 0..source_coord.len() {
            coord[self.permutation[axis]] = source_coord[axis];
        }

        coord
    }
}
