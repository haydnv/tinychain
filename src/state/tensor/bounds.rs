use std::fmt;
use std::iter;
use std::ops;

use itertools::Itertools;
use num::integer::Integer;

use super::stream::{AxisIter, Coords};

#[derive(Clone)]
pub enum AxisBounds {
    At(u64),
    In(ops::Range<u64>, u64),
    Of(Vec<u64>),
}

impl AxisBounds {
    pub fn all(dim: u64) -> AxisBounds {
        AxisBounds::In(0..dim, 1)
    }
}

impl PartialEq for AxisBounds {
    fn eq(&self, other: &AxisBounds) -> bool {
        use AxisBounds::*;
        match (self, other) {
            (At(l), At(r)) if l == r => true,
            (In(lr, ls), In(rr, rs)) if lr == rr && ls == rs => true,
            (Of(l), Of(r)) if l == r => true,
            _ => false,
        }
    }
}

impl From<u64> for AxisBounds {
    fn from(at: u64) -> AxisBounds {
        AxisBounds::At(at)
    }
}

impl From<Vec<u64>> for AxisBounds {
    fn from(of: Vec<u64>) -> AxisBounds {
        AxisBounds::Of(of)
    }
}

impl From<(ops::Range<u64>, u64)> for AxisBounds {
    fn from(slice: (ops::Range<u64>, u64)) -> AxisBounds {
        AxisBounds::In(slice.0, slice.1)
    }
}

impl fmt::Display for AxisBounds {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use AxisBounds::*;
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
pub struct Bounds {
    pub axes: Vec<AxisBounds>,
}

impl Bounds {
    pub fn all(shape: &Shape) -> Bounds {
        shape
            .0
            .iter()
            .map(|dim| AxisBounds::In(0..*dim, 1))
            .collect::<Vec<AxisBounds>>()
            .into()
    }

    pub fn affected(&self) -> Coords {
        use AxisBounds::*;
        let mut axes = Vec::with_capacity(self.len());
        for axis in 0..self.len() {
            axes.push(match &self[axis] {
                At(i) => AxisIter::One(iter::once(*i)),
                In(range, step) => AxisIter::Step(range.clone().step_by(*step as usize)),
                Of(indices) => AxisIter::Each(indices.to_vec(), 0),
            });
        }

        axes.iter().cloned().multi_cartesian_product()
    }

    pub fn into_coord(self) -> Vec<u64> {
        use AxisBounds::*;
        self.axes
            .into_iter()
            .map(|ab| match ab {
                At(x) => x,
                _ => panic!("Expected axis index but found AxisBound!"),
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.axes.len()
    }

    pub fn ndim(&self) -> usize {
        let mut ndim = 0;
        use AxisBounds::*;
        for axis in &self.axes {
            match axis {
                At(_) => {}
                _ => ndim += 1,
            }
        }
        ndim
    }

    pub fn normalize(&mut self, shape: &Shape) {
        assert!(self.len() <= shape.len());

        for axis in self.axes.len()..shape.len() {
            self.axes.push(AxisBounds::all(shape[axis]))
        }
    }

    pub fn to_vec(&self) -> Vec<AxisBounds> {
        self.axes.to_vec()
    }
}

impl PartialEq for Bounds {
    fn eq(&self, other: &Bounds) -> bool {
        if self.len() != other.len() {
            return false;
        }

        for axis in 0..self.len() {
            if self[axis] != other[axis] {
                return false;
            }
        }

        true
    }
}

impl Eq for Bounds {}

impl<Idx: std::slice::SliceIndex<[AxisBounds]>> ops::Index<Idx> for Bounds {
    type Output = Idx::Output;

    fn index(&self, i: Idx) -> &Self::Output {
        &self.axes[i]
    }
}

impl<Idx: std::slice::SliceIndex<[AxisBounds]>> ops::IndexMut<Idx> for Bounds {
    fn index_mut(&mut self, i: Idx) -> &mut Self::Output {
        &mut self.axes[i]
    }
}

impl From<Vec<AxisBounds>> for Bounds {
    fn from(axes: Vec<AxisBounds>) -> Bounds {
        Bounds { axes }
    }
}

impl From<&[u64]> for Bounds {
    fn from(coord: &[u64]) -> Bounds {
        let axes = coord.iter().map(|i| AxisBounds::At(*i)).collect();
        Bounds { axes }
    }
}

impl From<Vec<u64>> for Bounds {
    fn from(coord: Vec<u64>) -> Bounds {
        let axes = coord.iter().map(|i| AxisBounds::At(*i)).collect();
        Bounds { axes }
    }
}

impl From<(Vec<u64>, Vec<u64>)> for Bounds {
    fn from(bounds: (Vec<u64>, Vec<u64>)) -> Bounds {
        bounds
            .0
            .iter()
            .zip(bounds.1.iter())
            .map(|(s, e)| AxisBounds::In(*s..*e, 1))
            .collect::<Vec<AxisBounds>>()
            .into()
    }
}

impl From<(AxisBounds, Vec<u64>)> for Bounds {
    fn from(mut tuple: (AxisBounds, Vec<u64>)) -> Bounds {
        let mut axes = Vec::with_capacity(tuple.1.len() + 1);
        axes.push(tuple.0);
        for axis in tuple.1.drain(..) {
            axes.push(axis.into());
        }
        Bounds { axes }
    }
}

impl fmt::Display for Bounds {
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

#[derive(Clone, Default)]
pub struct Shape(Vec<u64>);

impl Shape {
    pub fn all(&self) -> Bounds {
        let mut axes = Vec::with_capacity(self.len());
        for dim in &self.0 {
            axes.push(AxisBounds::In(0..*dim, 1));
        }
        axes.into()
    }

    pub fn contains(&self, coord: &Bounds) -> bool {
        if coord.len() > self.len() {
            return false;
        }

        for axis in 0..coord.len() {
            let size = &self[axis];
            match &coord[axis] {
                AxisBounds::At(i) => {
                    if i > size {
                        return false;
                    }
                }
                AxisBounds::In(range, _) => {
                    if range.start > *size || range.end > *size {
                        return false;
                    }
                }
                AxisBounds::Of(indices) => {
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

    pub fn selection(&self, coord: &Bounds) -> Shape {
        assert!(self.contains(coord));

        let mut shape = Vec::with_capacity(self.len());
        for axis in 0..coord.len() {
            match &coord[axis] {
                AxisBounds::At(_) => {}
                AxisBounds::In(range, step) => {
                    let dim = (range.end - range.start).div_ceil(&step);
                    shape.push(dim)
                }
                AxisBounds::Of(indices) => shape.push(indices.len() as u64),
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

    pub fn remove(&mut self, axis: usize) {
        self.0.remove(axis);
    }

    pub fn size(&self) -> u64 {
        self.0.iter().product()
    }

    pub fn to_vec(&self) -> Vec<u64> {
        self.0.to_vec()
    }
}

impl PartialEq for Shape {
    fn eq(&self, other: &Shape) -> bool {
        self.0 == other.0
    }
}

impl Eq for Shape {}

impl<Idx: std::slice::SliceIndex<[u64]>> ops::Index<Idx> for Shape {
    type Output = Idx::Output;

    fn index(&self, i: Idx) -> &Self::Output {
        &self.0[i]
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
