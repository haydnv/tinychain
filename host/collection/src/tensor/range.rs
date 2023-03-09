use std::{fmt, iter, ops};

use safecast::{Match, TryCastFrom, TryCastInto};

use tc_error::*;
use tc_value::Value;

use super::Coord;

/// The range of a `Tensor` along a single axis.
#[derive(Clone, Eq, PartialEq)]
pub enum AxisRange {
    At(u64),
    In(ops::Range<u64>),
    Of(Vec<u64>),
}

impl AxisRange {
    /// `AxisRange` covering an entire axis
    pub fn all(dim: u64) -> AxisRange {
        AxisRange::In(0..dim)
    }

    /// The length of this range
    pub fn dim(&self) -> u64 {
        match self {
            Self::At(_) => 1,
            Self::In(range) => range.end - range.start,
            Self::Of(indices) => indices.len() as u64,
        }
    }

    /// Return `true` if this `AxisRange` specify a single index.
    pub fn is_index(&self) -> bool {
        if let Self::At(_) = self {
            true
        } else {
            false
        }
    }
}

impl From<u64> for AxisRange {
    fn from(at: u64) -> AxisRange {
        AxisRange::At(at)
    }
}

impl From<Vec<u64>> for AxisRange {
    fn from(of: Vec<u64>) -> AxisRange {
        AxisRange::Of(of)
    }
}

impl From<ops::Range<u64>> for AxisRange {
    fn from(range: ops::Range<u64>) -> AxisRange {
        AxisRange::In(range)
    }
}

impl TryCastFrom<Value> for AxisRange {
    fn can_cast_from(value: &Value) -> bool {
        value.matches::<u64>() || value.matches::<(u64, u64)>() || value.matches::<Vec<u64>>()
    }

    fn opt_cast_from(value: Value) -> Option<AxisRange> {
        if value.matches::<u64>() {
            value.opt_cast_into().map(AxisRange::At)
        } else if value.matches::<(u64, u64)>() {
            let range: (u64, u64) = value.opt_cast_into().unwrap();
            Some(AxisRange::In(range.0..range.1))
        } else if value.matches::<Vec<u64>>() {
            value.opt_cast_into().map(AxisRange::Of)
        } else {
            None
        }
    }
}

impl fmt::Debug for AxisRange {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for AxisRange {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use AxisRange::*;
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

/// `Tensor` range
#[derive(Clone)]
pub struct Range {
    axes: Vec<AxisRange>,
}

impl Range {
    /// The range of the entire `Tensor` with the given `Shape`
    pub fn all(shape: &Shape) -> Range {
        shape
            .0
            .iter()
            .map(|dim| AxisRange::In(0..*dim))
            .collect::<Vec<AxisRange>>()
            .into()
    }

    /// Return `true` if this `range` contain the given coordinate.
    pub fn contains_coord(&self, coord: &[u64]) -> bool {
        if coord.len() != self.len() {
            return false;
        }

        use AxisRange::*;
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

    /// Return `Some(Coord)` if this range matches a single `Coord`, otherwise `None`
    pub fn as_coord(&self, shape: &[u64]) -> Option<Coord> {
        if shape.len() != self.axes.len() {
            return None;
        }

        let mut coord = Vec::with_capacity(self.axes.len());
        for x in &self.axes {
            match x {
                AxisRange::At(i) => coord.push(*i),
                AxisRange::In(range) if range.end - range.start == 1 => coord.push(range.start),
                AxisRange::Of(indices) if indices.len() == 1 => coord.push(indices[0]),
                _ => return None,
            }
        }

        Some(coord)
    }

    /// Return `true` if this `Range` consists of `shape.len()` number of `AxisRange::At` items.
    pub fn is_coord(&self, shape: &[u64]) -> bool {
        self.len() == shape.len() && self.axes.iter().all(|bound| bound.is_index())
    }

    /// Return the number of dimensions of this `Tensor`.
    pub fn ndim(&self) -> usize {
        self.axes.iter().filter(|bound| !bound.is_index()).count()
    }

    /// Expand this `Range` to the entire given [`Shape`].
    ///
    /// Example:
    /// ```
    /// # use tc_tensor::{Range, Shape};
    /// let mut range = Range::from(&[0u64][..]);
    /// assert_eq!(range.to_shape(&Shape::from(vec![2, 3, 4])).unwrap(), Shape::from(vec![3, 4]));
    /// ```
    pub fn normalize(&mut self, shape: &Shape) {
        assert!(self.len() <= shape.len());

        for axis in self.axes.len()..shape.len() {
            self.axes.push(AxisRange::all(shape[axis]))
        }
    }

    /// Return the [`Shape`] of the `Tensor` slice with this `Range`.
    pub fn to_shape(&self, source_shape: &Shape) -> TCResult<Shape> {
        if source_shape.len() < self.len() {
            return Err(bad_request!(
                "invalid range {} for shape {}",
                self,
                source_shape
            ));
        }

        let mut shape = source_shape.to_vec();

        let mut axis = 0;
        for bound in &self.axes {
            match bound {
                AxisRange::In(range) => {
                    shape[axis] = range.end - range.start;
                    axis += 1;
                }
                AxisRange::At(_) => {
                    shape.remove(axis);
                }
                AxisRange::Of(indices) => {
                    shape[axis] = indices.len() as u64;
                    axis += 1;
                }
            }
        }

        Ok(shape.into())
    }

    /// Return the size of the slice with this `Range`,
    /// assuming they are of the same length as the source shape.
    pub fn size(&self) -> u64 {
        self.axes.iter().map(|bound| bound.dim()).product()
    }
}

impl IntoIterator for Range {
    type Item = AxisRange;
    type IntoIter = <Vec<AxisRange> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.axes.into_iter()
    }
}

impl Default for Range {
    fn default() -> Self {
        Self { axes: vec![] }
    }
}

impl ops::Deref for Range {
    type Target = Vec<AxisRange>;

    fn deref(&'_ self) -> &'_ Self::Target {
        &self.axes
    }
}

impl ops::DerefMut for Range {
    fn deref_mut(&'_ mut self) -> &'_ mut Self::Target {
        &mut self.axes
    }
}

impl PartialEq for Range {
    fn eq(&self, other: &Self) -> bool {
        self.axes == other.axes
    }
}

impl FromIterator<AxisRange> for Range {
    fn from_iter<T: IntoIterator<Item = AxisRange>>(iter: T) -> Self {
        Self {
            axes: iter.into_iter().collect(),
        }
    }
}

impl From<Vec<AxisRange>> for Range {
    fn from(axes: Vec<AxisRange>) -> Range {
        Range { axes }
    }
}

impl From<&[u64]> for Range {
    fn from(coord: &[u64]) -> Range {
        let axes = coord.iter().map(|i| AxisRange::At(*i)).collect();
        Range { axes }
    }
}

impl From<Vec<u64>> for Range {
    fn from(coord: Vec<u64>) -> Range {
        let axes = coord.iter().map(|i| AxisRange::At(*i)).collect();
        Range { axes }
    }
}

impl From<(Vec<u64>, Vec<u64>)> for Range {
    fn from(range: (Vec<u64>, Vec<u64>)) -> Range {
        range
            .0
            .iter()
            .zip(range.1.iter())
            .map(|(s, e)| AxisRange::In(*s..*e))
            .collect::<Vec<AxisRange>>()
            .into()
    }
}

impl From<(AxisRange, Vec<u64>)> for Range {
    fn from(tuple: (AxisRange, Vec<u64>)) -> Range {
        let mut axes = Vec::with_capacity(tuple.1.len() + 1);
        axes.push(tuple.0);
        for axis in tuple.1.into_iter() {
            axes.push(axis.into());
        }
        Range { axes }
    }
}

impl TryCastFrom<Value> for Range {
    fn can_cast_from(value: &Value) -> bool {
        value.matches::<Vec<AxisRange>>()
    }

    fn opt_cast_from(value: Value) -> Option<Range> {
        let range: Option<Vec<AxisRange>> = value.opt_cast_into();
        range.map(Range::from)
    }
}

impl fmt::Debug for Range {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("[")?;

        let len = self.axes.len();
        for i in 0..self.axes.len() {
            write!(f, "{:?}", self.axes[i])?;

            if i < (len - 1) {
                f.write_str(", ")?;
            }
        }

        f.write_str("]")
    }
}

impl fmt::Display for Range {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("[")?;

        let len = self.axes.len();
        for i in 0..self.axes.len() {
            write!(f, "{}", self.axes[i])?;

            if i < (len - 1) {
                f.write_str(", ")?;
            }
        }

        f.write_str("]")
    }
}

/// The shape of a `Tensor`
#[derive(Clone, Default, Eq, PartialEq)]
pub struct Shape(Vec<u64>);

impl Shape {
    /// Return true if the given [`Range`] fit within this `Shape`.
    pub fn contains_range(&self, range: &Range) -> bool {
        if range.len() > self.len() {
            return false;
        }

        for axis in 0..range.len() {
            let size = &self[axis];
            match &range[axis] {
                AxisRange::At(i) => {
                    if i > size {
                        return false;
                    }
                }
                AxisRange::In(range) => {
                    if range.start > *size || range.end > *size {
                        return false;
                    }
                }
                AxisRange::Of(indices) => {
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
        self.0.iter().product()
    }

    /// Return a `TCError` if this `Shape` is empty.
    pub fn validate(&self, debug_info: &'static str) -> TCResult<()> {
        if self.0.is_empty() {
            return Err(bad_request!(
                "error in {}: invalid tensor shape {}",
                debug_info,
                self
            ));
        }

        let mut size = 1u64;
        for dim in &self.0 {
            if dim == &0 {
                return Err(bad_request!(
                    "error in {}: invalid tensor dimension {}",
                    debug_info,
                    dim
                ));
            } else if let Some(m) = size.checked_mul(*dim) {
                size = m;
            } else {
                return Err(bad_request!(
                    "error in {}: tensor shape {} exceeds the maximum allowed size of 2^64",
                    debug_info,
                    self
                ));
            }
        }

        Ok(())
    }

    /// Return a `TCError` if any of the given axes is out of range.
    pub fn validate_axes(&self, axes: &[usize]) -> TCResult<()> {
        match axes.iter().max() {
            Some(max) if max > &self.len() => {
                Err(bad_request!("shape {} has no axis {}", self, max))
            }
            _ => Ok(()),
        }
    }

    /// Return a `TCError` if the given `Range` don't fit within this `Shape`.
    pub fn validate_range(&self, range: &Range) -> TCResult<()> {
        if self.contains_range(range) {
            Ok(())
        } else {
            Err(bad_request!(
                "Tensor of shape {} does not contain range {}",
                self,
                range
            ))
        }
    }

    /// Return a `TCError` if the given `coord` doesn't fit within this `Shape`.
    pub fn validate_coord(&self, coord: &[u64]) -> TCResult<()> {
        for (axis, index) in coord.iter().enumerate() {
            if index >= &self[axis] {
                return Err(bad_request!(
                    "Tensor of shape {} does not contain {}",
                    self,
                    Value::from_iter(coord.to_vec())
                ));
            }
        }

        Ok(())
    }
}

impl ops::Deref for Shape {
    type Target = Vec<u64>;

    fn deref(&'_ self) -> &'_ Vec<u64> {
        &self.0
    }
}

impl ops::DerefMut for Shape {
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
