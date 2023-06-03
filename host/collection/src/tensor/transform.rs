use std::collections::{HashMap, HashSet};
use std::iter;

use log::warn;

use tc_error::*;

use super::shape::{AxisRange, Range, Shape};
use super::{strides_for, Axes, Coord, Strides};

#[derive(Clone)]
pub struct Broadcast {
    source_shape: Shape,
    shape: Shape,
    broadcast: Vec<bool>,
    offset: usize,
}

impl Broadcast {
    pub fn new(source_shape: Shape, shape: Shape) -> TCResult<Broadcast> {
        source_shape.validate()?;
        shape.validate()?;

        if source_shape.is_empty() {
            return Err(bad_request!("cannot broadcast an empty Tensor"));
        } else if shape.is_empty() {
            return Err(bad_request!("cannot broadcast into an empty Tensor"));
        } else if source_shape == shape {
            warn!(
                "broadcast a Tensor with shape {:?} into {:?}",
                source_shape, shape
            );
        }

        let ndim = shape.len();
        debug_assert!(source_shape.len() <= ndim);

        if source_shape.len() > ndim {
            return Err(bad_request!(
                "cannot broadcast {:?} into {:?}",
                source_shape,
                shape
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
                return Err(bad_request!(
                    "cannot broadcast {:?} into {:?}",
                    source_shape,
                    shape
                ));
            }
        }

        Ok(Broadcast {
            source_shape,
            shape,
            broadcast,
            offset,
        })
    }

    pub fn shape(&'_ self) -> &'_ Shape {
        &self.shape
    }

    pub fn invert_range(&self, range: Range) -> Range {
        let source_ndim = self.source_shape.len();
        let mut source_range = Vec::with_capacity(source_ndim);

        for axis in 0..source_ndim {
            if axis + self.offset < range.len() {
                if self.broadcast[axis + self.offset] {
                    if range[axis + self.offset].is_index() {
                        source_range.push(AxisRange::from(0))
                    } else {
                        source_range.push(AxisRange::In(0..1, 1))
                    }
                } else {
                    source_range.push(range[axis + self.offset].clone())
                }
            } else {
                source_range.push(AxisRange::In(0..self.source_shape[axis], 1))
            }
        }

        if source_range.iter().all(|bound| bound.is_index()) {
            // can't broadcast a slice with shape []
            if let Some(AxisRange::At(i)) = source_range.pop() {
                source_range.push(AxisRange::In(i..i + 1, 1));
            }
        }

        Range::from(source_range)
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
}

#[derive(Clone)]
pub struct Expand {
    source_shape: Shape,
    shape: Shape,
    expand: Axes,
}

impl Expand {
    pub fn new(source_shape: Shape, expand: Axes) -> TCResult<Expand> {
        source_shape.validate_axes(&expand)?;

        let mut shape = source_shape.to_vec();
        for x in expand.iter().rev().copied() {
            shape.insert(x, 1);
        }

        Ok(Expand {
            source_shape,
            shape: shape.into(),
            expand,
        })
    }

    #[inline]
    pub fn expand_axes(&self) -> &[usize] {
        &self.expand
    }

    pub fn shape(&'_ self) -> &'_ Shape {
        &self.shape
    }

    pub fn invert_range(&self, mut range: Range) -> Range {
        assert_eq!(range.len(), self.shape.len());

        for x in self.expand.iter().rev().copied() {
            if x < range.len() {
                let removed = range.remove(x);
                if !removed.is_index() {
                    if x == range.len() {
                        let bound = match range.pop().unwrap() {
                            AxisRange::At(i) => AxisRange::In(i..i + 1, 1),
                            other => other,
                        };

                        range.push(bound);
                    } else {
                        let bound = match range.remove(x) {
                            AxisRange::At(i) => AxisRange::In(i..i + 1, 1),
                            other => other,
                        };

                        range.insert(x, bound);
                    }
                }
            }
        }

        range
    }

    pub fn invert_coord(&self, mut coord: Coord) -> Coord {
        debug_assert_eq!(coord.len(), self.shape.len());

        for x in self.expand.iter().rev().copied() {
            debug_assert_eq!(coord[x], 0);
            coord.remove(x);
        }

        coord
    }
}

#[derive(Clone)]
pub struct Reduce {
    source_shape: Shape,
    axes: Vec<usize>,
    shape: Shape,
}

impl Reduce {
    pub fn new(source_shape: Shape, axes: Vec<usize>, keepdims: bool) -> TCResult<Reduce> {
        source_shape.validate_axes(&axes)?;

        let mut shape = source_shape.clone();

        if keepdims {
            for x in axes.iter().copied() {
                shape[x] = 1;
            }
        } else {
            for x in axes.iter().rev().copied() {
                shape.remove(x);
            }
        }

        Ok(Reduce {
            source_shape,
            shape,
            axes,
        })
    }

    pub fn shape(&'_ self) -> &'_ Shape {
        &self.shape
    }

    pub fn invert_range(&self, mut range: Range) -> Range {
        if self.shape.len() == self.source_shape.len() {
            for x in self.axes.iter().copied() {
                if x < range.len() {
                    range[x] = AxisRange::all(self.source_shape[x]);
                }
            }
        } else {
            for x in self.axes.iter().rev().copied() {
                if x <= range.len() {
                    range.insert(x, AxisRange::all(self.source_shape[x]));
                }
            }
        }

        range
    }

    pub fn invert_coord(&self, coord: &[u64]) -> Range {
        let mut range: Vec<AxisRange> = coord.iter().map(|i| AxisRange::At(*i)).collect();

        if self.shape.len() == self.source_shape.len() {
            for x in self.axes.iter().copied() {
                range[x] = AxisRange::all(self.source_shape[x]);
            }
        } else {
            for x in self.axes.iter().rev().copied() {
                range.insert(x, AxisRange::all(self.source_shape[x]));
            }
        }

        range.into()
    }

    pub fn reduce_axes(&self) -> &[usize] {
        &self.axes
    }
}

#[derive(Clone)]
pub struct Reshape {
    source_shape: Shape,
    source_strides: Strides,

    shape: Shape,
    strides: Strides,
}

impl Reshape {
    pub fn new(source_shape: Shape, shape: Shape) -> TCResult<Self> {
        source_shape.validate()?;
        shape.validate()?;

        if source_shape.size() != shape.size() {
            return Err(bad_request!(
                "cannot reshape tensor with shape {:?} into {:?}",
                source_shape,
                shape
            ));
        }

        let source_strides = strides_for(&source_shape, source_shape.len());
        let strides = strides_for(&shape, shape.len());

        Ok(Self {
            source_shape,
            source_strides,
            shape,
            strides,
        })
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn invert_coord(&self, coord: Coord) -> Coord {
        assert_eq!(coord.len(), self.shape.len());

        let offset: u64 = coord
            .into_iter()
            .zip(&self.strides)
            .map(|(x, stride)| x * stride)
            .sum();

        self.source_strides
            .iter()
            .map(|stride| offset / stride)
            .zip(self.source_shape.iter())
            .map(|(axis_offset, dim)| axis_offset % dim)
            .collect()
    }
}

#[derive(Clone)]
pub struct Slice {
    source_shape: Shape,
    shape: Shape,
    range: Range,
    offset: HashMap<usize, u64>,
    elided: HashMap<usize, u64>,
}

impl Slice {
    pub fn new(source_shape: Shape, range: Range) -> TCResult<Slice> {
        source_shape.validate_range(&range)?;

        let mut shape: Coord = Vec::with_capacity(source_shape.len());
        let mut offset = HashMap::new();
        let mut elided = HashMap::new();

        for axis in 0..range.len() {
            match &range[axis] {
                AxisRange::At(c) => {
                    elided.insert(axis, *c);
                }
                AxisRange::In(range, step) => {
                    let dim = (range.end - range.start) / step;
                    shape.push(dim);
                    offset.insert(axis, range.start);
                }
                AxisRange::Of(indices) => {
                    shape.push(indices.len() as u64);
                }
            }
        }

        for axis in range.len()..source_shape.len() {
            shape.push(source_shape[axis]);
        }

        let shape: Shape = shape.into();

        Ok(Slice {
            source_shape,
            shape,
            range,
            offset,
            elided,
        })
    }

    pub fn range(&'_ self) -> &'_ Range {
        &self.range
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

    pub fn invert_range(&self, mut range: Range) -> Range {
        let range = range.normalize(&self.shape);

        if range.is_empty() || range == Range::all(self.shape()) {
            return self.range.clone();
        }

        let mut source_range = Vec::with_capacity(self.source_shape.len());
        let mut source_axis = 0;
        let mut axis = 0;
        while source_axis < self.source_shape.len() {
            if let Some(c) = self.elided.get(&source_axis) {
                source_axis += 1;
                source_range.push(AxisRange::At(*c));
                continue;
            }

            match &range[axis] {
                AxisRange::In(range, step) => {
                    if source_axis < self.range.len() {
                        if let AxisRange::In(source_axis_range, source_step) =
                            &self.range[source_axis]
                        {
                            let start = range.start + source_axis_range.start;
                            let end = start + (range.end - range.start);
                            let step = step * source_step;
                            source_range.push(AxisRange::In(start..end, step));
                        } else {
                            assert_eq!(range.start, 0);
                            source_range.push(self.range[source_axis].clone());
                        }
                    } else {
                        source_range.push(AxisRange::In(range.clone(), *step));
                    }
                }
                AxisRange::Of(indices) => {
                    let offset = self.offset.get(&source_axis).unwrap_or(&0);
                    source_range.push(
                        indices
                            .iter()
                            .map(|i| i + offset)
                            .collect::<Vec<u64>>()
                            .into(),
                    )
                }
                AxisRange::At(i) => {
                    let offset = self.offset.get(&source_axis).unwrap_or(&0);
                    source_range.push((i + offset).into())
                }
            }

            source_axis += 1;
            axis += 1;
        }

        source_range.into()
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
}

#[derive(Clone)]
pub struct Transpose {
    source_shape: Shape,
    shape: Shape,
    permutation: Axes,
    inverse_permutation: Axes,
}

impl Transpose {
    pub fn new(source_shape: Shape, permutation: Option<Vec<usize>>) -> TCResult<Transpose> {
        let ndim = source_shape.len();

        let permutation = if let Some(permutation) = permutation {
            permutation
        } else {
            (0..ndim).rev().collect()
        };

        source_shape.validate_axes(&permutation)?;

        if permutation.len() != ndim {
            return Err(bad_request!(
                "tensor with shape {:?} cannot transpose axes {:?}",
                source_shape,
                permutation
            ));
        } else if permutation.iter().max().expect("transpose last axis") > &ndim {
            return Err(bad_request!(
                "shape {:?} has no axis {}",
                source_shape,
                permutation.iter().max().unwrap()
            ));
        } else if permutation.iter().cloned().collect::<HashSet<_>>().len() != permutation.len() {
            return Err(bad_request!(
                "cannot transpose the same axis twice: {:?}",
                permutation
            ));
        }

        let mut shape = Coord::with_capacity(ndim);
        for axis in permutation.iter().copied() {
            shape.push(source_shape[axis]);
        }

        let shape = Shape::from(shape);

        let mut inverse_permutation = vec![0; ndim];
        for (i, x) in permutation.iter().copied().enumerate() {
            inverse_permutation[x] = i;
        }

        Ok(Transpose {
            source_shape,
            shape,
            permutation,
            inverse_permutation,
        })
    }

    pub fn shape(&'_ self) -> &'_ Shape {
        &self.shape
    }

    pub fn invert_range(&self, range: &Range) -> Range {
        let mut source_range = Range::all(&self.source_shape);
        for axis in 0..range.len() {
            source_range[self.permutation[axis]] = range[axis].clone();
        }
        source_range
    }

    pub fn invert_coord(&self, coord: Coord) -> Coord {
        assert_eq!(coord.len(), self.permutation.len());

        let mut source_coord = vec![0; coord.len()];
        for (x, i) in coord.into_iter().enumerate() {
            source_coord[self.permutation[x]] = i;
        }

        source_coord
    }
}
