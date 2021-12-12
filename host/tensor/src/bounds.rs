use std::fmt;
use std::iter::{self, FromIterator};
use std::ops::{self, Deref, DerefMut};

use itertools::{Itertools, MultiProduct};
use safecast::{Match, TryCastFrom, TryCastInto};

use tc_error::*;
use tc_value::Value;

use super::Coord;

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

/// The bounds of a `Tensor` along a single axis.
#[derive(Clone)]
pub enum AxisBounds {
    At(u64),
    In(ops::Range<u64>),
    Of(Vec<u64>),
}

impl AxisBounds {
    /// `AxisBounds` covering an entire axis
    pub fn all(dim: u64) -> AxisBounds {
        AxisBounds::In(0..dim)
    }

    /// The length of these bounds
    pub fn dim(&self) -> u64 {
        match self {
            Self::At(_) => 1,
            Self::In(range) => range.end - range.start,
            Self::Of(indices) => indices.len() as u64,
        }
    }

    /// Return `true` if these `AxisBounds` specify a single index.
    pub fn is_index(&self) -> bool {
        if let Self::At(_) = self {
            true
        } else {
            false
        }
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

impl fmt::Debug for AxisBounds {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
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
                "{{{}}}",
                indices
                    .iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<String>>()
                    .join(", ")
            ),
        }
    }
}

/// `Tensor` bounds
#[derive(Clone)]
pub struct Bounds {
    axes: Vec<AxisBounds>,
}

impl Bounds {
    /// The bounds of the entire `Tensor` with the given `Shape`
    pub fn all(shape: &Shape) -> Bounds {
        shape
            .0
            .iter()
            .map(|dim| AxisBounds::In(0..*dim))
            .collect::<Vec<AxisBounds>>()
            .into()
    }

    /// Return an iterator over all the [`Coord`]s within these `Bounds`.
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

    /// Return `true` if these `bounds` contain the given coordinate.
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

    /// Return `Some(Coord)` if these bounds match a single `Coord`, otherwise `None`
    pub fn as_coord(&self, shape: &[u64]) -> Option<Coord> {
        if shape.len() != self.axes.len() {
            return None;
        }

        let mut coord = Vec::with_capacity(self.axes.len());
        for x in &self.axes {
            match x {
                AxisBounds::At(i) => coord.push(*i),
                AxisBounds::In(range) if range.end - range.start == 1 => coord.push(range.start),
                AxisBounds::Of(indices) if indices.len() == 1 => coord.push(indices[0]),
                _ => return None,
            }
        }

        Some(coord)
    }

    /// Return `true` if these `Bounds` consist of `shape.len()` number of `AxisBounds::At` items.
    pub fn is_coord(&self, shape: &[u64]) -> bool {
        self.len() == shape.len() && self.axes.iter().all(|bound| bound.is_index())
    }

    pub fn ndim(&self) -> usize {
        self.axes.iter().filter(|bound| !bound.is_index()).count()
    }

    /// Expand these `Bounds` to the entire given [`Shape`].
    ///
    /// Example:
    /// ```
    /// # use tc_tensor::{Bounds, Shape};
    /// let mut bounds = Bounds::from(&[0u64][..]);
    /// assert_eq!(bounds.to_shape(&Shape::from(vec![2, 3, 4])).unwrap(), Shape::from(vec![3, 4]));
    /// ```
    pub fn normalize(&mut self, shape: &Shape) {
        assert!(self.len() <= shape.len());

        for axis in self.axes.len()..shape.len() {
            self.axes.push(AxisBounds::all(shape[axis]))
        }
    }

    /// Return the [`Shape`] of the `Tensor` slice with these `Bounds`.
    pub fn to_shape(&self, source_shape: &Shape) -> TCResult<Shape> {
        if source_shape.len() < self.len() {
            return Err(TCError::unsupported(format!(
                "invalid bounds {} for shape {}",
                self, source_shape
            )));
        }

        let mut shape = source_shape.to_vec();

        let mut axis = 0;
        for bound in &self.axes {
            match bound {
                AxisBounds::In(range) => {
                    shape[axis] = range.end - range.start;
                    axis += 1;
                }
                AxisBounds::At(_) => {
                    shape.remove(axis);
                }
                AxisBounds::Of(indices) => {
                    shape[axis] = indices.len() as u64;
                    axis += 1;
                }
            }
        }

        Ok(shape.into())
    }

    /// Return the size of the slice with these `Bounds`,
    /// assuming they are of the same length as the source shape.
    pub fn size(&self) -> u64 {
        self.axes.iter().map(|bound| bound.dim()).product()
    }
}

impl IntoIterator for Bounds {
    type Item = AxisBounds;
    type IntoIter = <Vec<AxisBounds> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.axes.into_iter()
    }
}

impl Default for Bounds {
    fn default() -> Self {
        Self { axes: vec![] }
    }
}

impl Deref for Bounds {
    type Target = Vec<AxisBounds>;

    fn deref(&'_ self) -> &'_ Self::Target {
        &self.axes
    }
}

impl DerefMut for Bounds {
    fn deref_mut(&'_ mut self) -> &'_ mut Self::Target {
        &mut self.axes
    }
}

impl PartialEq for Bounds {
    fn eq(&self, other: &Self) -> bool {
        self.axes == other.axes
    }
}

impl FromIterator<AxisBounds> for Bounds {
    fn from_iter<T: IntoIterator<Item = AxisBounds>>(iter: T) -> Self {
        Self {
            axes: iter.into_iter().collect(),
        }
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

impl fmt::Debug for Bounds {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "[{}]",
            self.axes
                .iter()
                .map(|axis| format!("{:?}", axis))
                .collect::<Vec<String>>()
                .join(", ")
        )
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

/// The shape of a `Tensor`
#[derive(Clone, Default, Eq, PartialEq)]
pub struct Shape(Vec<u64>);

impl Shape {
    /// Return true if the given [`Bounds`] fit within this `Shape`.
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

    /// Return `true` if the given `coord` exists within this `Shape`.
    pub fn contains_coord(&self, coord: &[u64]) -> bool {
        if coord.len() != self.len() {
            return false;
        }

        for axis in 0..coord.len() {
            if coord[axis] >= self[axis] {
                return false;
            }
        }

        true
    }

    /// Consume this `Shape` and return the underlying `Vec<u64>`.
    pub fn into_vec(self) -> Vec<u64> {
        self.0
    }

    /// Return the origin [`Coord`] of this `Shape`.
    pub fn origin(&self) -> Coord {
        iter::repeat(0).take(self.len()).collect()
    }

    /// Return the number of elements contained within this `Shape`.
    pub fn size(&self) -> u64 {
        log::debug!("size of {}?", self);
        self.0.iter().product()
    }

    /// Return a `TCError` if this `Shape` is empty.
    pub fn validate(&self) -> TCResult<()> {
        if self.0.is_empty() {
            return Err(TCError::bad_request("invalid tensor shape", self));
        }

        let mut size = 1u64;
        for dim in &self.0 {
            if dim == &0 {
                return Err(TCError::bad_request("invalid tensor dimension", dim));
            } else if let Some(m) = size.checked_mul(*dim) {
                size = m;
            } else {
                return Err(TCError::bad_request(
                    "tensor shape exceeds the maximum allowed size of 2^64",
                    self,
                ));
            }
        }

        Ok(())
    }

    /// Return a `TCError` if any of the given axes is out of bounds.
    pub fn validate_axes(&self, axes: &[usize]) -> TCResult<()> {
        match axes.iter().max() {
            Some(max) if max > &self.len() => Err(TCError::unsupported(format!(
                "shape {} has no axis {}",
                self, max
            ))),
            _ => Ok(()),
        }
    }

    /// Return a `TCError` if the given `Bounds` don't fit within this `Shape`.
    pub fn validate_bounds(&self, bounds: &Bounds) -> TCResult<()> {
        if self.contains_bounds(bounds) {
            Ok(())
        } else {
            Err(TCError::unsupported(format!(
                "Tensor of shape {} does not contain bounds {}",
                self, bounds
            )))
        }
    }

    /// Return a `TCError` if the given `coord` doesn't fit within this `Shape`.
    pub fn validate_coord(&self, coord: &[u64]) -> TCResult<()> {
        for (axis, index) in coord.iter().enumerate() {
            if index >= &self[axis] {
                return Err(TCError::unsupported(format!(
                    "Tensor of shape {} does not contain {}",
                    self,
                    Value::from_iter(coord.to_vec())
                )));
            }
        }

        Ok(())
    }
}

impl Deref for Shape {
    type Target = Vec<u64>;

    fn deref(&'_ self) -> &'_ Vec<u64> {
        &self.0
    }
}

impl DerefMut for Shape {
    fn deref_mut(&'_ mut self) -> &'_ mut Vec<u64> {
        &mut self.0
    }
}

impl From<Vec<u64>> for Shape {
    fn from(shape: Vec<u64>) -> Shape {
        Shape(shape)
    }
}

impl FromIterator<u64> for Shape {
    fn from_iter<I: IntoIterator<Item = u64>>(iter: I) -> Self {
        let dims = Vec::<u64>::from_iter(iter);
        Self(dims)
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

impl fmt::Debug for Shape {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
