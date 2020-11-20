use std::convert::TryFrom;
use std::fmt;
use std::iter;
use std::ops;

use itertools::{Itertools, MultiProduct};

use crate::class::TCResult;
use crate::error;
use crate::scalar::{ScalarInstance, TryCastFrom, TryCastInto, Value};

pub type Coords = MultiProduct<AxisIter>;

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

#[derive(Clone)]
pub enum AxisBounds {
    At(u64),
    In(ops::Range<u64>),
    Of(Vec<u64>),
}

impl AxisBounds {
    pub fn all(dim: u64) -> AxisBounds {
        AxisBounds::In(0..dim)
    }
}

impl PartialEq for AxisBounds {
    fn eq(&self, other: &AxisBounds) -> bool {
        use AxisBounds::*;
        match (self, other) {
            (At(l), At(r)) if l == r => true,
            (In(lr), In(rr)) if lr == rr => true,
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

impl From<ops::Range<u64>> for AxisBounds {
    fn from(range: ops::Range<u64>) -> AxisBounds {
        AxisBounds::In(range)
    }
}

impl TryCastFrom<Value> for AxisBounds {
    fn can_cast_from(value: &Value) -> bool {
        value.matches::<u64>() || value.matches::<(u64, u64)>() || value.matches::<Vec<u64>>()
    }

    fn opt_cast_from(value: Value) -> Option<AxisBounds> {
        if value.matches::<u64>() {
            value.opt_cast_into().map(AxisBounds::At)
        } else if value.matches::<(u64, u64)>() {
            let range: (u64, u64) = value.opt_cast_into().unwrap();
            Some(AxisBounds::In(range.0..range.1))
        } else if value.matches::<Vec<u64>>() {
            value.opt_cast_into().map(AxisBounds::Of)
        } else {
            None
        }
    }
}

impl fmt::Display for AxisBounds {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use AxisBounds::*;
        match self {
            At(at) => write!(f, "{}", at),
            In(range) => write!(f, "[{}, {})", range.start, range.end),
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
            .map(|dim| AxisBounds::In(0..*dim))
            .collect::<Vec<AxisBounds>>()
            .into()
    }

    pub fn affected(&self) -> Coords {
        use AxisBounds::*;
        let mut axes = Vec::with_capacity(self.len());
        for axis in 0..self.len() {
            axes.push(match &self[axis] {
                At(i) => AxisIter::One(iter::once(*i)),
                In(range) => AxisIter::Step(range.clone().step_by(1)),
                Of(indices) => AxisIter::Each(indices.to_vec(), 0),
            });
        }

        axes.iter().cloned().multi_cartesian_product()
    }

    pub fn contains_coord(&self, coord: &[u64]) -> bool {
        if coord.len() != self.len() {
            return false;
        }

        use AxisBounds::*;
        for (bound, c) in self.axes.iter().zip(coord) {
            match bound {
                At(i) if i != c => return false,
                In(range) if !range.contains(c) => return false,
                Of(indices) if !indices.contains(c) => return false,
                _ => {}
            }
        }

        true
    }

    pub fn is_coord(&self) -> bool {
        self.axes.iter().all(|bound| match bound {
            AxisBounds::At(_) => true,
            _ => false,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.axes.is_empty()
    }

    pub fn iter(&'_ self) -> impl Iterator<Item = (usize, &'_ AxisBounds)> {
        self.axes.iter().enumerate()
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

    pub fn remove(&mut self, axis: usize) {
        self.axes.remove(axis);
    }

    pub fn size(&self) -> u64 {
        if self.is_empty() {
            return 0;
        }

        let mut size = 1;
        for bound in &self.axes {
            match bound {
                AxisBounds::At(_) => {}
                AxisBounds::In(range) => size *= range.end - range.start,
                AxisBounds::Of(indices) => size *= indices.len() as u64,
            }
        }

        size
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
            .map(|(s, e)| AxisBounds::In(*s..*e))
            .collect::<Vec<AxisBounds>>()
            .into()
    }
}

impl From<(AxisBounds, Vec<u64>)> for Bounds {
    fn from(tuple: (AxisBounds, Vec<u64>)) -> Bounds {
        let mut axes = Vec::with_capacity(tuple.1.len() + 1);
        axes.push(tuple.0);
        for axis in tuple.1.into_iter() {
            axes.push(axis.into());
        }
        Bounds { axes }
    }
}

impl TryCastFrom<Value> for Bounds {
    fn can_cast_from(value: &Value) -> bool {
        value.matches::<Vec<AxisBounds>>()
    }

    fn opt_cast_from(value: Value) -> Option<Bounds> {
        let bounds: Option<Vec<AxisBounds>> = value.opt_cast_into();
        bounds.map(Bounds::from)
    }
}

impl TryFrom<Bounds> for Vec<u64> {
    type Error = error::TCError;

    fn try_from(bounds: Bounds) -> TCResult<Vec<u64>> {
        let mut coord = Vec::with_capacity(bounds.len());
        for bound in bounds.axes.into_iter() {
            match bound {
                AxisBounds::At(x) => coord.push(x),
                other => {
                    return Err(error::bad_request("Not a coordinate", other));
                }
            }
        }

        Ok(coord)
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
            axes.push(AxisBounds::In(0..*dim));
        }
        axes.into()
    }

    pub fn contains_bounds(&self, bounds: &Bounds) -> bool {
        if bounds.len() > self.len() {
            return false;
        }

        for axis in 0..bounds.len() {
            let size = &self[axis];
            match &bounds[axis] {
                AxisBounds::At(i) => {
                    if i > size {
                        return false;
                    }
                }
                AxisBounds::In(range) => {
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

    pub fn contains_coord(&self, coord: &[u64]) -> bool {
        if coord.len() != self.len() {
            return false;
        }

        for axis in 0..coord.len() {
            if coord[axis] > self[axis] {
                return false;
            }
        }

        true
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn remove(&mut self, axis: usize) -> u64 {
        self.0.remove(axis)
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

impl TryCastFrom<Value> for Shape {
    fn can_cast_from(value: &Value) -> bool {
        value.matches::<Vec<u64>>()
    }

    fn opt_cast_from(value: Value) -> Option<Shape> {
        let shape: Option<Vec<u64>> = value.opt_cast_into();
        shape.map(Shape::from)
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
