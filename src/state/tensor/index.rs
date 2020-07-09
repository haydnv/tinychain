use std::fmt;
use std::iter;
use std::ops;

use itertools::{Itertools, MultiProduct};
use num::integer::Integer;

pub type Coords = MultiProduct<AxisIter>;

#[derive(Clone)]
pub enum AxisIndex {
    At(u64),
    In(ops::Range<u64>, u64),
    Of(Vec<u64>),
}

impl AxisIndex {
    pub fn all(dim: u64) -> AxisIndex {
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
    pub axes: Vec<AxisIndex>,
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

    pub fn affected(&self) -> Coords {
        use AxisIndex::*;
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

impl From<Vec<u64>> for Index {
    fn from(coord: Vec<u64>) -> Index {
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
    pub fn all(&self) -> Index {
        let mut axes = Vec::with_capacity(self.len());
        for dim in &self.0 {
            axes.push(AxisIndex::In(0..*dim, 1));
        }
        axes.into()
    }

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

    pub fn selection(&self, coord: &Index) -> Shape {
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

impl PartialEq for Shape {
    fn eq(&self, other: &Shape) -> bool {
        self.0 == other.0
    }
}

impl Eq for Shape {}

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

#[derive(Clone)]
pub enum AxisIter {
    One(std::iter::Once<u64>),
    Each(Vec<u64>, usize),
    Step(iter::StepBy<ops::Range<u64>>),
}

impl Iterator for AxisIter {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        use AxisIter::*;
        match self {
            One(iter) => iter.next(),
            Each(v, at) => {
                if at == &v.len() {
                    None
                } else {
                    Some(v[*at])
                }
            }
            Step(iter) => iter.next(),
        }
    }
}
