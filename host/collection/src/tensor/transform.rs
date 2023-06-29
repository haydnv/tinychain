use std::iter;

use log::warn;

use tc_error::*;

use super::shape::{AxisRange, Range, Shape};
use super::{coord_of, strides_for, Axes, Coord, Strides};

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
    shape: Shape,
    expand: Axes,
}

impl Expand {
    pub fn new(source_shape: Shape, mut expand: Axes) -> TCResult<Expand> {
        source_shape.validate_axes(&expand)?;
        expand.sort();

        let mut shape = source_shape.to_vec();
        for x in expand.iter().rev().copied() {
            shape.insert(x, 1);
        }

        Ok(Expand {
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
    pub fn new(source_shape: Shape, mut axes: Vec<usize>, keepdims: bool) -> TCResult<Reduce> {
        source_shape.validate_axes(&axes)?;

        axes.sort();

        for i in 0..(axes.len() - 1) {
            if axes[i] == axes[i + 1] {
                return Err(bad_request!(
                    "cannot reduce dimension {dim} more than once",
                    dim = axes[i]
                ));
            }
        }

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

    pub fn keepdims(&self) -> bool {
        self.shape.len() == self.source_shape.len()
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn invert_axes(&self, axes: Vec<usize>) -> Vec<usize> {
        let mut source_axes: Axes = (0..self.source_shape.len()).into_iter().collect();
        for x in self.axes.iter().rev().copied() {
            source_axes.remove(x);
        }

        axes.into_iter().map(|x| source_axes[x]).collect()
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
        let mut range = Vec::with_capacity(self.source_shape.len());
        range.extend(coord.iter().copied().map(|i| AxisRange::At(i)));

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

    pub fn strides(&self) -> &[u64] {
        &self.strides
    }

    pub fn source_strides(&self) -> &[u64] {
        &self.source_strides
    }

    pub fn invert_coord(&self, coord: Coord) -> Coord {
        assert_eq!(coord.len(), self.shape.len());

        let offset: u64 = coord
            .into_iter()
            .zip(&self.strides)
            .map(|(x, stride)| x * stride)
            .sum();

        coord_of(offset, &self.source_strides, &self.source_shape)
    }
}

#[derive(Clone)]
pub struct Slice {
    source_shape: Shape,
    shape: Shape,
    range: Range,
}

impl Slice {
    pub fn new(source_shape: Shape, range: Range) -> TCResult<Slice> {
        source_shape.validate_range(&range)?;

        if range.is_coord(source_shape.as_slice()) {
            return Err(bad_request!(
                "slice {range:?} of {source_shape:?} has no size"
            ));
        }

        let mut shape = Vec::with_capacity(source_shape.len());

        for bound in range.iter() {
            match bound {
                AxisRange::At(_) => {} // no-op
                AxisRange::In(range, step) => {
                    let dim = (range.end - range.start) / step;
                    shape.push(dim);
                }
                AxisRange::Of(indices) => {
                    shape.push(indices.len() as u64);
                }
            }
        }

        shape.extend_from_slice(&source_shape[range.len()..]);

        let shape = Shape::from(shape);

        Ok(Slice {
            source_shape,
            shape,
            range,
        })
    }

    pub fn range(&self) -> &Range {
        &self.range
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn source_shape(&self) -> &Shape {
        &self.source_shape
    }

    pub fn invert_range(&self, range: Range) -> Range {
        let range = if range.is_empty() || range == Range::all(self.shape()) {
            return self.range.clone();
        } else {
            range.normalize(&self.shape)
        };

        let mut source_range = Vec::with_capacity(self.shape.len());
        let mut axis = 0;

        for axis_range in self.range.iter() {
            let axis_range = match axis_range {
                AxisRange::At(i) => AxisRange::At(*i),
                AxisRange::In(source_range, source_step) => match &range[axis] {
                    AxisRange::At(i) => {
                        debug_assert!(source_range.start + (i * source_step) < source_range.end);
                        AxisRange::At(source_range.start + (i * source_step))
                    }
                    AxisRange::In(axis_range, step) => {
                        debug_assert!(source_range.start + axis_range.start <= source_range.end);
                        debug_assert!(source_range.start + axis_range.end <= source_range.end);

                        let (source_start, source_end, source_step) = (
                            axis_range.start + source_range.start,
                            axis_range.end + source_range.start,
                            step * source_step,
                        );

                        AxisRange::In(source_start..source_end, source_step)
                    }
                    AxisRange::Of(indices) => {
                        let indices = indices
                            .iter()
                            .copied()
                            .map(|i| source_range.start + i)
                            .collect::<Vec<u64>>();

                        debug_assert!(indices.iter().copied().all(|i| i < source_range.end));

                        AxisRange::Of(indices)
                    }
                },
                AxisRange::Of(source_indices) => match &range[axis] {
                    AxisRange::At(i) => AxisRange::At(source_indices[*i as usize]),
                    AxisRange::In(axis_range, step) => {
                        debug_assert!(axis_range.start as usize <= source_indices.len());
                        debug_assert!(axis_range.end as usize <= source_indices.len());

                        let indices = source_indices
                            [(axis_range.start as usize)..(axis_range.end as usize)]
                            .iter()
                            .step_by(*step as usize)
                            .copied()
                            .collect();

                        AxisRange::Of(indices)
                    }
                    AxisRange::Of(indices) => {
                        let indices = indices
                            .iter()
                            .copied()
                            .map(|i| source_indices[i as usize])
                            .collect();

                        AxisRange::Of(indices)
                    }
                },
            };

            if !axis_range.is_index() {
                axis += 1;
            }

            source_range.push(axis_range);
        }

        source_range.extend(self.range.iter().skip(range.len()).cloned());

        source_range.into()
    }

    pub fn invert_coord(&self, coord: Coord) -> Coord {
        let mut source_coord = Coord::with_capacity(self.range.len() + coord.len());

        let mut axis = 0;
        for range in self.range.iter() {
            match range {
                AxisRange::At(i) => source_coord.push(*i),
                AxisRange::In(range, step) => {
                    let i = range.start + (coord[axis] * step);
                    source_coord.push(i);
                    axis += 1;
                }
                AxisRange::Of(indices) => {
                    source_coord.push(indices[coord[axis] as usize]);
                    axis += 1;
                }
            }
        }

        source_coord.extend(coord.into_iter().skip(source_coord.len()));

        source_coord
    }
}

#[derive(Clone)]
pub struct Transpose {
    source_shape: Shape,
    shape: Shape,
    permutation: Axes,
}

impl Transpose {
    pub fn new(source_shape: Shape, permutation: Option<Vec<usize>>) -> TCResult<Transpose> {
        let ndim = source_shape.len();

        let permutation = if let Some(axes) = permutation {
            if axes.len() == source_shape.len()
                && (0..source_shape.len())
                    .into_iter()
                    .all(|x| axes.contains(&x))
            {
                Ok(axes)
            } else {
                Err(bad_request!(
                    "invalid permutation for shape {:?}: {:?}",
                    source_shape,
                    axes
                ))
            }
        } else {
            Ok((0..source_shape.len()).into_iter().rev().collect())
        }?;

        let mut shape = Coord::with_capacity(ndim);
        for axis in permutation.iter().copied() {
            shape.push(source_shape[axis]);
        }

        Ok(Transpose {
            source_shape,
            shape: shape.into(),
            permutation,
        })
    }

    pub fn axes(&self) -> &[usize] {
        &self.permutation
    }

    pub fn shape(&self) -> &Shape {
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
