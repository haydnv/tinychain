use std::collections::HashMap;
use std::iter;
use std::ops;

use afarray::Coords;
use log::debug;

use tc_error::*;
use tcgeneric::Tuple;

use crate::bounds::{AxisBounds, Bounds, Shape};

use super::Coord;

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
        if source_shape.is_empty() {
            return Err(TCError::unsupported("cannot broadcast an empty Tensor"));
        } else if shape.is_empty() {
            return Err(TCError::unsupported(
                "cannot broadcast into an empty Tensor",
            ));
        }

        let ndim = shape.len();
        if source_shape.len() > ndim {
            return Err(TCError::unsupported(format!(
                "cannot broadcast {} into {}",
                source_shape, shape
            )));
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
                    &format!("cannot broadcast into {}", shape),
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

        let mut bounds = Bounds::all(self.shape());

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

    pub fn invert_bounds(&self, bounds: Bounds) -> Bounds {
        let source_ndim = self.source_shape.len();
        let mut source_bounds = Vec::with_capacity(source_ndim);

        for axis in 0..source_ndim {
            if axis + self.offset < bounds.len() {
                if self.broadcast[axis + self.offset] {
                    if bounds[axis + self.offset].is_index() {
                        source_bounds.push(AxisBounds::from(0))
                    } else {
                        source_bounds.push(AxisBounds::In(0..1))
                    }
                } else {
                    source_bounds.push(bounds[axis + self.offset].clone())
                }
            } else {
                source_bounds.push(AxisBounds::In(0..self.source_shape[axis]))
            }
        }

        if source_bounds.iter().all(|bound| bound.is_index()) {
            // can't broadcast a slice with shape []
            if let Some(AxisBounds::At(i)) = source_bounds.pop() {
                source_bounds.push(AxisBounds::In(i..i + 1));
            }
        }

        Bounds::from(source_bounds)
    }

    pub fn invert_coord(&self, coord: &[u64]) -> Coord {
        debug_assert_eq!(coord.len(), self.shape.len());

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

    pub fn invert_coords(&self, coords: &Coords) -> Coords {
        assert_eq!(coords.ndim(), self.shape.len());
        coords.unbroadcast(&self.source_shape, &self.broadcast)
    }

    pub fn map_coord(&self, coord: Coord) -> Bounds {
        self.map_bounds(coord.into())
    }
}

#[derive(Clone)]
pub struct Expand {
    source_shape: Shape,
    shape: Shape,
    expand: usize,
}

impl Expand {
    pub fn new(source_shape: Shape, expand: usize) -> TCResult<Expand> {
        if expand > source_shape.len() {
            return Err(TCError::bad_request("axis out of bounds", expand));
        }

        let mut shape = source_shape.to_vec();
        shape.insert(expand, 1);

        Ok(Expand {
            source_shape,
            shape: shape.into(),
            expand,
        })
    }

    #[inline]
    pub fn expand_axis(&self) -> usize {
        self.expand
    }

    pub fn invert_axes(&self, axes: Vec<usize>) -> Vec<usize> {
        axes.into_iter()
            .filter_map(|x| {
                if x == self.expand {
                    None
                } else if x > self.expand {
                    Some(x - 1)
                } else {
                    Some(x)
                }
            })
            .collect()
    }

    pub fn invert_axis(&self, bounds: &Bounds) -> Option<usize> {
        debug!("invert expand axis {} in bounds {}", self.expand, bounds);
        assert!(self.expand < bounds.len());

        if bounds[self.expand].is_index() {
            return None;
        }

        let mut expand = self.expand;
        for bound in &bounds[..self.expand] {
            if bound.is_index() {
                expand -= 1;
            }
        }

        debug!("bound at expansion index is {:?}", bounds.get(expand));

        Some(expand)
    }

    pub fn shape(&'_ self) -> &'_ Shape {
        &self.shape
    }

    pub fn invert_bounds(&self, mut bounds: Bounds) -> Bounds {
        assert_eq!(bounds.len(), self.shape.len());

        if self.expand < bounds.len() {
            let removed = bounds.remove(self.expand);
            if !removed.is_index() {
                if self.expand == bounds.len() {
                    let bound = match bounds.pop().unwrap() {
                        AxisBounds::At(i) => AxisBounds::In(i..i + 1),
                        other => other,
                    };

                    bounds.push(bound);
                } else {
                    let bound = match bounds.remove(self.expand) {
                        AxisBounds::At(i) => AxisBounds::In(i..i + 1),
                        other => other,
                    };

                    bounds.insert(self.expand, bound);
                }
            }
        }

        bounds
    }

    pub fn invert_coord(&self, coord: &[u64]) -> Coord {
        debug_assert_eq!(coord.len(), self.shape.len());

        let mut inverted = Vec::with_capacity(self.source_shape.len());
        inverted.extend(&coord[..self.expand]);

        if self.expand < self.source_shape.len() {
            inverted.extend(&coord[self.expand + 1..]);
        }

        inverted
    }

    pub fn invert_coords(&self, coords: &Coords) -> Coords {
        coords.contract_dim(self.expand)
    }

    pub fn map_coord(&self, mut coord: Coord) -> Coord {
        debug_assert_eq!(coord.len(), self.source_shape.len());
        coord.insert(self.expand, 0);
        coord
    }
}

#[derive(Clone)]
pub struct Flip {
    shape: Shape,
    axis: usize,
}

impl Flip {
    pub fn new(shape: Shape, axis: usize) -> TCResult<Self> {
        if axis > shape.len() {
            Err(TCError::unsupported(format!(
                "invalid axis {} for shape {}",
                axis, shape
            )))
        } else {
            Ok(Self { shape, axis })
        }
    }

    pub fn axis(&self) -> usize {
        self.axis
    }

    pub fn flip_bounds(&self, mut bounds: Bounds) -> Bounds {
        if bounds.len() < self.axis {
            return bounds;
        }

        let dim = self.shape[self.axis];
        bounds[self.axis] = match &bounds[self.axis] {
            AxisBounds::At(i) => AxisBounds::At(dim - i),
            AxisBounds::In(ops::Range { start, end }) => AxisBounds::In((dim - end)..(dim - start)),
            AxisBounds::Of(indices) => {
                AxisBounds::Of(indices.into_iter().map(|i| dim - i).collect())
            }
        };

        bounds
    }

    pub fn flip_coord(&self, mut coord: Coord) -> Coord {
        assert_eq!(coord.len(), self.shape.len());
        coord[self.axis] = self.shape[self.axis] - coord[self.axis];
        coord
    }

    pub fn flip_coords(&self, coords: Coords) -> Coords {
        coords.flip(&self.shape, self.axis)
    }

    pub fn invert_axis(&self, bounds: &Bounds) -> Option<usize> {
        if bounds.len() <= self.axis {
            None
        } else if bounds[self.axis].is_index() {
            None
        } else {
            let elided = bounds[..self.axis]
                .iter()
                .filter(|bound| bound.is_index())
                .count();
            Some(self.axis - elided)
        }
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
            return Err(TCError::unsupported(format!(
                "cannot reduce axis {} of tensor with shape {}",
                axis, source_shape
            )));
        }

        let mut shape = source_shape.clone();
        shape.remove(axis);

        Ok(Reduce {
            source_shape,
            shape,
            axis,
        })
    }

    pub fn shape(&'_ self) -> &'_ Shape {
        &self.shape
    }

    pub fn invert_axes(&self, axes: Vec<usize>) -> Vec<usize> {
        axes.into_iter()
            .map(|x| if x >= self.axis { x + 1 } else { x })
            .collect()
    }

    pub fn invert_axis(&self, bounds: &Bounds) -> usize {
        let elided = bounds[..self.axis]
            .iter()
            .filter(|bound| bound.is_index())
            .count();

        self.axis - elided
    }

    pub fn invert_bounds(&self, mut bounds: Bounds) -> Bounds {
        if bounds.len() < self.axis {
            bounds
        } else {
            bounds.insert(self.axis, AxisBounds::all(self.source_shape[self.axis]));
            bounds
        }
    }

    pub fn invert_coord(&self, coord: &[u64]) -> Bounds {
        let mut bounds: Vec<AxisBounds> = coord.iter().map(|i| AxisBounds::At(*i)).collect();
        bounds.insert(self.axis, AxisBounds::all(self.source_shape[self.axis]));
        bounds.into()
    }

    pub fn reduce_axis(&self) -> usize {
        self.axis
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

        let mut shape: Coord = Vec::with_capacity(source_shape.len());
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

    pub fn invert_bounds(&self, mut bounds: Bounds) -> Bounds {
        bounds.normalize(&self.shape);

        if bounds.is_empty() || bounds == Bounds::all(self.shape()) {
            return self.bounds.clone();
        }

        let mut source_bounds = Vec::with_capacity(self.source_shape.len());
        let mut source_axis = 0;
        let mut axis = 0;
        while source_axis < self.source_shape.len() {
            if let Some(c) = self.elided.get(&source_axis) {
                source_axis += 1;
                source_bounds.push(AxisBounds::At(*c));
                continue;
            }

            use AxisBounds::*;
            match &bounds[axis] {
                In(range) => {
                    if source_axis < self.bounds.len() {
                        if let In(source_range) = &self.bounds[source_axis] {
                            let start = range.start + source_range.start;
                            let end = start + (range.end - range.start);
                            source_bounds.push((start..end).into());
                        } else {
                            assert_eq!(range.start, 0);
                            source_bounds.push(self.bounds[source_axis].clone());
                        }
                    } else {
                        source_bounds.push(In(range.clone()));
                    }
                }
                Of(indices) => {
                    let offset = self.offset.get(&source_axis).unwrap_or(&0);
                    source_bounds.push(indices.iter().map(|i| i + offset).collect::<Coord>().into())
                }
                At(i) => {
                    let offset = self.offset.get(&source_axis).unwrap_or(&0);
                    source_bounds.push((i + offset).into())
                }
            }

            source_axis += 1;
            axis += 1;
        }

        source_bounds.into()
    }

    pub fn invert_coord(&self, coord: &[u64]) -> Coord {
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

    pub fn invert_coords(&self, coords: &Coords) -> Coords {
        let source_coords = coords.unslice(&self.source_shape, &self.elided, &self.offset);
        source_coords
    }

    pub fn map_coord(&self, source_coord: Coord) -> Coord {
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

    pub fn map_coords(&self, source_coords: Coords) -> Coords {
        assert_eq!(source_coords.ndim(), self.source_shape.len());
        source_coords.slice(&self.shape, &self.elided, &self.offset)
    }
}

#[derive(Clone)]
pub struct Transpose {
    source_shape: Shape,
    shape: Shape,
    permutation: Vec<usize>,
    inverse_permutation: Vec<usize>,
}

impl Transpose {
    pub fn new(source_shape: Shape, permutation: Option<Vec<usize>>) -> TCResult<Transpose> {
        let ndim = source_shape.len();
        let permutation = permutation.unwrap_or_else(|| (0..ndim).rev().collect());

        if permutation.len() != ndim {
            return Err(TCError::unsupported(format!(
                "tensor with shape {} cannot transpose axes {}",
                source_shape,
                Tuple::from(permutation)
            )));
        } else if let Some(max_axis) = permutation.iter().max() {
            if max_axis >= &ndim {
                return Err(TCError::bad_request(
                    "cannot transpose nonexistent axis",
                    max_axis,
                ));
            }
        }

        let mut shape: Coord = Vec::with_capacity(ndim);
        for axis in &permutation {
            shape.push(source_shape[*axis]);
        }
        let shape: Shape = shape.into();

        let mut inverse_permutation = vec![0; ndim];
        for (i, x) in permutation.iter().enumerate() {
            inverse_permutation[*x] = i;
        }

        Ok(Transpose {
            source_shape,
            shape,
            permutation,
            inverse_permutation,
        })
    }

    pub fn invert_axes(&self, axes: Vec<usize>) -> Vec<usize> {
        axes.into_iter().map(|x| self.permutation[x]).collect()
    }

    pub fn map_axes(&self, axes: &[usize]) -> Vec<usize> {
        axes.into_iter()
            .map(|x| self.inverse_permutation[*x])
            .collect()
    }

    pub fn shape(&'_ self) -> &'_ Shape {
        &self.shape
    }

    pub fn invert_bounds(&self, bounds: &Bounds) -> Bounds {
        let mut source_bounds = Bounds::all(&self.source_shape);
        for axis in 0..bounds.len() {
            source_bounds[self.permutation[axis]] = bounds[axis].clone();
        }
        source_bounds
    }

    pub fn invert_coord(&self, coord: &[u64]) -> Coord {
        assert_eq!(coord.len(), self.permutation.len());

        let mut source_coord = vec![0; coord.len()];
        for axis in 0..coord.len() {
            source_coord[self.permutation[axis]] = coord[axis];
        }

        source_coord
    }

    pub fn invert_coords(&self, coords: &Coords) -> Coords {
        assert_eq!(coords.ndim(), self.permutation.len());
        coords.transpose(Some(&self.inverse_permutation))
    }

    pub fn invert_permutation(&self, bounds: &Bounds) -> Vec<usize> {
        debug!(
            "source permutation is {:?}, bounds are {}",
            self.permutation, bounds
        );

        let mut offset = 0;
        let mut offsets = Vec::with_capacity(self.shape.len());
        for bound in self.invert_bounds(bounds).iter() {
            if bound.is_index() {
                offset += 1;
            }

            offsets.push(offset);
        }

        let mut permutation = Vec::with_capacity(self.permutation.len());
        for (i, bound) in bounds.iter().enumerate() {
            if bound.is_index() {
                // pass
            } else {
                permutation.push(self.permutation[i] - offsets[self.permutation[i]]);
            }
        }

        for i in bounds.len()..self.shape.len() {
            permutation.push(self.permutation[i] - offsets[self.permutation[i]]);
        }

        permutation
    }

    pub fn map_coords(&self, coords: Coords) -> Coords {
        assert_eq!(coords.ndim(), self.permutation.len());
        coords.transpose(Some(&self.permutation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_invert_bounds() {
        let shape = Shape::from(vec![2, 3, 4, 1]);
        let rebase = Broadcast::new(shape.clone(), vec![5, 6, 2, 3, 4, 10].into()).unwrap();
        assert_eq!(
            rebase.invert_bounds(vec![AxisBounds::In(0..1)].into()),
            Bounds::all(&shape)
        )
    }

    #[test]
    fn test_slice_invert_bounds() {
        let rebase = Slice::new(vec![2, 3, 4, 5].into(), Bounds::from(vec![0])).unwrap();
        assert_eq!(rebase.shape().to_vec(), vec![3, 4, 5]);
        assert_eq!(
            rebase.invert_bounds(Bounds::from(vec![
                AxisBounds::In(0..3),
                AxisBounds::In(0..4),
                AxisBounds::At(1)
            ])),
            Bounds::from(vec![
                AxisBounds::At(0),
                AxisBounds::In(0..3),
                AxisBounds::In(0..4),
                AxisBounds::At(1)
            ])
        );

        let rebase = Slice::new(
            vec![2, 3, 4, 5].into(),
            Bounds::from(vec![
                AxisBounds::At(0),
                AxisBounds::In(0..3),
                AxisBounds::In(1..3),
                AxisBounds::At(1),
            ]),
        )
        .unwrap();

        assert_eq!(
            rebase.invert_bounds(Bounds::from(vec![AxisBounds::At(0), AxisBounds::In(0..2)])),
            Bounds::from(vec![
                AxisBounds::At(0),
                AxisBounds::At(0),
                AxisBounds::In(1..3),
                AxisBounds::At(1)
            ])
        );
    }

    #[test]
    fn test_transpose_invert_permutation() {
        let rebase = Transpose::new(vec![10, 15, 20].into(), Some(vec![0, 1, 2])).unwrap();
        assert_eq!(
            rebase.invert_permutation(&Bounds::from(vec![AxisBounds::At(0), AxisBounds::In(2..5)])),
            vec![0, 1]
        );

        let rebase = Transpose::new(vec![10, 15, 20].into(), None).unwrap();
        assert_eq!(
            rebase.invert_permutation(&Bounds::from(vec![AxisBounds::At(0), AxisBounds::In(2..5)])),
            vec![1, 0]
        );
        assert_eq!(
            rebase.invert_permutation(&Bounds::from(vec![AxisBounds::In(0..2), AxisBounds::At(1)])),
            vec![1, 0]
        );
        assert_eq!(
            rebase.invert_permutation(&Bounds::from(vec![AxisBounds::In(0..2)])),
            vec![2, 1, 0]
        );

        let rebase = Transpose::new(vec![10, 15, 20, 25].into(), None).unwrap();
        assert_eq!(
            rebase.invert_permutation(&Bounds::from(vec![AxisBounds::In(0..2), AxisBounds::At(1)])),
            vec![2, 1, 0]
        );

        let rebase = Transpose::new(vec![10, 15, 20, 25].into(), Some(vec![3, 0, 1, 2])).unwrap();
        assert_eq!(
            rebase.invert_permutation(&Bounds::from(vec![AxisBounds::In(0..2), AxisBounds::At(1)])),
            vec![2, 0, 1]
        );
    }
}
