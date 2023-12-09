use std::{fmt, iter, ops};

use async_hash::{Digest, Hash, Output};
use async_trait::async_trait;
use b_table::collate::{Collate, Collator, Overlap, OverlapsRange, OverlapsValue};
use destream::{de, en};
use futures::TryFutureExt;
use itertools::{Itertools, MultiProduct};
use safecast::{CastFrom, CastInto, Match, TryCastFrom, TryCastInto};
use smallvec::{smallvec, SmallVec};

use tc_error::*;
use tc_value::Value;
use tcgeneric::Tuple;

use super::Coord;

#[derive(Clone)]
pub enum AxisRangeIter {
    At(iter::Once<u64>),
    In(iter::StepBy<ops::Range<u64>>),
    Of(smallvec::IntoIter<[u64; 32]>),
}

impl Iterator for AxisRangeIter {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::At(iter) => iter.next(),
            Self::In(iter) => iter.next(),
            Self::Of(iter) => iter.next(),
        }
    }
}

/// The range of a `Tensor` along a single axis.
#[derive(Clone, Eq, PartialEq)]
pub enum AxisRange {
    At(u64),
    In(ops::Range<u64>, u64),
    Of(SmallVec<[u64; 32]>),
}

impl AxisRange {
    /// `AxisRange` covering an entire axis
    pub fn all(dim: u64) -> AxisRange {
        AxisRange::In(0..dim, 1)
    }

    /// The length of this range
    pub fn dim(&self) -> u64 {
        match self {
            Self::At(_) => 1,
            Self::In(range, step) => (range.end - range.start) / step,
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

    #[inline]
    fn to_range(&self) -> std::ops::Range<u64> {
        let start = match self {
            Self::At(i) => *i,
            Self::In(range, _step) => range.start,
            Self::Of(indices) => *indices.iter().min().expect("min"),
        };

        let end = match self {
            Self::At(i) => i + 1,
            Self::In(range, _step) => range.end,
            Self::Of(indices) => *indices.iter().max().expect("max"),
        };

        start..end
    }
}

impl OverlapsRange<AxisRange, Collator<u64>> for AxisRange {
    fn overlaps(&self, other: &AxisRange, collator: &Collator<u64>) -> Overlap {
        let this = self.to_range();
        let that = other.to_range();
        this.overlaps(&that, collator)
    }
}

impl OverlapsValue<u64, Collator<u64>> for AxisRange {
    fn overlaps_value(&self, value: &u64, collator: &Collator<u64>) -> Overlap {
        match self {
            Self::At(this) => collator.cmp(this, value).into(),
            Self::In(this, _step) => this.overlaps_value(value, collator),
            Self::Of(this) if this.is_empty() => Overlap::Narrow,
            Self::Of(this) => to_range(this).overlaps_value(value, collator),
        }
    }
}

impl IntoIterator for AxisRange {
    type Item = u64;
    type IntoIter = AxisRangeIter;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            Self::At(i) => AxisRangeIter::At(iter::once(i)),
            Self::In(range, step) => AxisRangeIter::In(range.into_iter().step_by(step as usize)),
            Self::Of(indices) => AxisRangeIter::Of(indices.into_iter()),
        }
    }
}

impl From<u64> for AxisRange {
    fn from(at: u64) -> AxisRange {
        AxisRange::At(at)
    }
}

impl From<SmallVec<[u64; 32]>> for AxisRange {
    fn from(of: SmallVec<[u64; 32]>) -> AxisRange {
        AxisRange::Of(of)
    }
}

impl From<ops::Range<u64>> for AxisRange {
    fn from(range: ops::Range<u64>) -> AxisRange {
        AxisRange::In(range, 1)
    }
}

impl TryFrom<AxisRange> for ha_ndarray::AxisRange {
    type Error = TCError;

    fn try_from(range: AxisRange) -> Result<Self, Self::Error> {
        match range {
            AxisRange::At(i) => i
                .try_into()
                .map(ha_ndarray::AxisRange::At)
                .map_err(|cause| bad_request!("bad range: {cause}")),

            AxisRange::In(range, step) => {
                let start = range
                    .start
                    .try_into()
                    .map_err(|cause| bad_request!("bad range start: {cause}"))?;

                let stop = range
                    .end
                    .try_into()
                    .map_err(|cause| bad_request!("bad range start: {cause}"))?;

                let step = step
                    .try_into()
                    .map_err(|cause| bad_request!("bad range start: {cause}"))?;

                Ok(ha_ndarray::AxisRange::In(start, stop, step))
            }

            AxisRange::Of(indices) => {
                let indices = indices
                    .into_iter()
                    .map(|i| {
                        i.try_into()
                            .map_err(|cause| bad_request!("bad index: {cause}"))
                    })
                    .collect::<Result<_, TCError>>()?;

                Ok(ha_ndarray::AxisRange::Of(indices))
            }
        }
    }
}

impl fmt::Debug for AxisRange {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use AxisRange::*;
        match self {
            At(at) => write!(f, "{at}"),
            In(range, 1) => write!(f, "[{}, {})", range.start, range.end),
            In(range, step) => write!(f, "[{}, {})/{}", range.start, range.end, step),
            Of(indices) if indices.is_empty() => f.write_str("{}"),
            Of(indices) => {
                f.write_str("{")?;

                for i in &indices[..indices.len() - 1] {
                    write!(f, "{i}, ")?;
                }

                write!(f, "{}}}", indices[indices.len() - 1])
            }
        }
    }
}

/// `Tensor` range
#[derive(Clone)]
pub struct Range {
    axes: SmallVec<[AxisRange; 8]>,
}

impl Range {
    /// The range of the entire `Tensor` with the given `Shape`
    pub fn all(shape: &Shape) -> Range {
        shape
            .0
            .iter()
            .copied()
            .map(|dim| AxisRange::In(0..dim, 1))
            .collect::<SmallVec<[AxisRange; 8]>>()
            .into()
    }

    /// Return an iterator over all the [`Coord`]s within this `Range`.
    pub fn affected(&self) -> MultiProduct<AxisRangeIter> {
        self.axes
            .iter()
            .cloned()
            .map(|axis_range| axis_range.into_iter())
            .multi_cartesian_product()
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
                In(range, step) if !range.contains(c) || c % step != 0 => return false,
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

        let mut coord = Coord::with_capacity(self.axes.len());
        for x in &self.axes {
            match x {
                AxisRange::At(i) => coord.push(*i),
                AxisRange::In(range, step)
                    if range.end > range.start && range.end - range.start <= *step =>
                {
                    coord.push(range.start)
                }
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
    /// # use smallvec::smallvec;
    /// # use tc_collection::tensor::{Range, Shape};
    /// let mut range = Range::from(&[0u64][..]);
    /// assert_eq!(
    ///     range.to_shape(&Shape::from(smallvec![2, 3, 4])).unwrap(),
    ///     Shape::from(smallvec![3, 4]));
    /// ```
    pub fn normalize(mut self, shape: &[u64]) -> Self {
        assert!(shape.len() >= self.len());

        self.axes.reserve(shape.len() - self.len());
        for dim in shape[self.len()..].iter().copied() {
            self.axes.push(AxisRange::all(dim))
        }

        self
    }

    /// Return the [`Shape`] of the `Tensor` slice with this `Range`.
    pub fn shape(&self) -> Shape {
        self.axes
            .iter()
            .filter_map(|bound| match bound {
                AxisRange::At(_) => None,
                AxisRange::In(range, step) => Some((range.end - range.start) / *step),
                AxisRange::Of(indices) => Some(indices.len() as u64),
            })
            .collect()
    }

    /// Return the size of the slice with this `Range`,
    /// assuming they are of the same length as the source shape.
    pub fn size(&self) -> u64 {
        self.axes.iter().map(|bound| bound.dim()).product()
    }
}

impl OverlapsRange<Range, Collator<u64>> for Range {
    fn overlaps(&self, other: &Range, collator: &Collator<u64>) -> Overlap {
        match (self.is_empty(), other.is_empty()) {
            (true, true) => return Overlap::Equal,
            (true, false) => return Overlap::Greater,
            (false, true) => return Overlap::Narrow,
            (false, false) => {}
        }

        let mut overlap = Overlap::Equal;
        for (this, that) in self.iter().zip(other.iter()) {
            match this.overlaps(that, collator) {
                Overlap::Less => return Overlap::Less,
                Overlap::Greater => return Overlap::Greater,
                axis_overlap => overlap = overlap.then(axis_overlap),
            }
        }

        overlap
    }
}

impl OverlapsValue<Coord, Collator<u64>> for Range {
    fn overlaps_value(&self, value: &Coord, collator: &Collator<u64>) -> Overlap {
        let mut overlap = if self.len() == value.len() {
            Overlap::Equal
        } else {
            Overlap::Wide
        };

        for (axis_bound, i) in self.iter().zip(value) {
            match axis_bound.overlaps_value(i, collator) {
                Overlap::Less => return Overlap::Less,
                Overlap::Greater => return Overlap::Greater,
                axis_overlap => overlap = overlap.then(axis_overlap),
            }
        }

        overlap
    }
}

impl IntoIterator for Range {
    type Item = AxisRange;
    type IntoIter = <SmallVec<[AxisRange; 8]> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.axes.into_iter()
    }
}

impl<'a> IntoIterator for &'a Range {
    type Item = &'a AxisRange;
    type IntoIter = <&'a [AxisRange] as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.axes.as_slice().iter()
    }
}

impl Default for Range {
    fn default() -> Self {
        Self { axes: smallvec![] }
    }
}

impl ops::Deref for Range {
    type Target = SmallVec<[AxisRange; 8]>;

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

impl From<SmallVec<[AxisRange; 8]>> for Range {
    fn from(axes: SmallVec<[AxisRange; 8]>) -> Self {
        Self { axes }
    }
}

impl From<&[u64]> for Range {
    fn from(coord: &[u64]) -> Range {
        Range {
            axes: coord.iter().map(|i| AxisRange::At(*i)).collect(),
        }
    }
}

impl From<Coord> for Range {
    fn from(coord: Coord) -> Range {
        Range {
            axes: coord.into_iter().map(AxisRange::At).collect(),
        }
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

/// The shape of a `Tensor`
#[derive(Clone, Default, Eq, PartialEq)]
pub struct Shape(SmallVec<[u64; 8]>);

impl Shape {
    /// Return true if the given [`Range`] fit within this [`Shape`].
    pub fn contains_range(&self, range: &Range) -> bool {
        if range.len() > self.len() {
            return false;
        }

        self.0
            .iter()
            .copied()
            .zip(range)
            .all(|(dim, bound)| match bound {
                AxisRange::At(i) => *i < dim,
                AxisRange::In(range, _step) => range.start < dim && range.end <= dim,
                AxisRange::Of(indices) => indices.iter().copied().all(|i| i < dim),
            })
    }

    /// Return true if the given [`Range`] matches this [`Shape`] exactly.
    pub fn is_covered_by(&self, range: &Range) -> bool {
        if range.len() > self.len() {
            return false;
        }

        range
            .iter()
            .zip(self)
            .all(|(axis_range, dim)| match axis_range {
                AxisRange::In(range, 1) if range.start == 0 && &range.end == dim => true,
                AxisRange::Of(indices) => indices
                    .iter()
                    .enumerate()
                    .all(|(i, idx)| (i as u64) == *idx),
                _ => false,
            })
    }

    /// Return `true` if the given `coord` exists within this `Shape`.
    pub fn contains_coord(&self, coord: &[u64]) -> bool {
        if coord.len() == self.len() {
            self.iter().zip(coord).all(|(dim, i)| i < dim)
        } else {
            false
        }
    }

    /// Consume this `Shape` and return the underlying `SmallVec<[u64; 8]>`.
    pub fn into_inner(self) -> SmallVec<[u64; 8]> {
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

    /// Return an error if this `Shape` is empty or oversized.
    pub fn validate(&self) -> Result<(), TCError> {
        if self.0.is_empty() {
            return Err(bad_request!("invalid shape {:?}", self));
        }

        let mut size = 1u64;
        for dim in self.0.iter().copied() {
            size = if dim == 0 || dim > u32::MAX as u64 {
                Err(bad_request!("invalid dimension: {}", dim))
            } else if let Some(size) = size.checked_mul(dim) {
                Ok(size)
            } else {
                Err(bad_request!(
                    "shape {:?} exceeds the maximum allowed size of 2^64",
                    self
                ))
            }?;
        }

        Ok(())
    }

    /// Return an error if any of the given axes is out of range.
    pub fn validate_axes(&self, axes: &[usize], require_ndim: bool) -> Result<(), TCError> {
        if let Some(max) = axes.iter().max().copied() {
            if max >= self.len() {
                #[cfg(debug_assertions)]
                panic!("shape {self:?} has no axis {max}");

                #[cfg(not(debug_assertions))]
                return Err(bad_request!("shape {self:?} has no axis {max}"));
            }
        }

        if (1..axes.len()).any(|i| axes[i..].contains(&axes[i - 1])) {
            return Err(bad_request!("{axes:?} contains a duplicate axis"));
        }

        if require_ndim {
            if !axes.is_empty() && axes.len() != self.len() {
                #[cfg(debug_assertions)]
                panic!("invalid permutation for {self:?}: {axes:?}");

                #[cfg(not(debug_assertions))]
                return Err(bad_request!("invalid permutation for {self:?}: {axes:?}"));
            }
        }

        Ok(())
    }

    /// Return an error if the given `Range` don't fit within this `Shape`.
    #[inline]
    pub fn validate_range(&self, range: &Range) -> Result<(), TCError> {
        if self.contains_range(range) {
            Ok(())
        } else {
            #[cfg(debug_assertions)]
            panic!("shape {self:?} does not contain {range:?}");

            #[cfg(not(debug_assertions))]
            Err(bad_request!("shape {self:?} does not contain {range:?}"))
        }
    }

    /// Return an error if the given `coord` doesn't fit within this `Shape`.
    #[inline]
    pub fn validate_coord(&self, coord: &[u64]) -> Result<(), TCError> {
        if self.contains_coord(coord) {
            Ok(())
        } else {
            #[cfg(debug_assertions)]
            panic!("{self:?} does not contain {coord:?}");

            #[cfg(not(debug_assertions))]
            Err(bad_request!("{self:?} does not contain {coord:?}"))
        }
    }
}

impl ops::Deref for Shape {
    type Target = SmallVec<[u64; 8]>;

    fn deref(&'_ self) -> &'_ Self::Target {
        &self.0
    }
}

impl ops::DerefMut for Shape {
    fn deref_mut(&'_ mut self) -> &'_ mut Self::Target {
        &mut self.0
    }
}

impl<D: Digest> Hash<D> for Shape {
    fn hash(self) -> Output<D> {
        Hash::<D>::hash(self.0)
    }
}

impl<'a, D: Digest> Hash<D> for &'a Shape {
    fn hash(self) -> Output<D> {
        let mut hasher = D::new();
        for dim in self.0.iter().copied() {
            let hash = Hash::<D>::hash(dim);
            hasher.update(hash);
        }
        hasher.finalize()
    }
}

impl IntoIterator for Shape {
    type Item = u64;
    type IntoIter = <SmallVec<[u64; 8]> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a Shape {
    type Item = &'a u64;
    type IntoIter = <&'a [u64] as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        (&self.0).into_iter()
    }
}

impl FromIterator<u64> for Shape {
    fn from_iter<I: IntoIterator<Item = u64>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl From<SmallVec<[u64; 8]>> for Shape {
    fn from(shape: SmallVec<[u64; 8]>) -> Self {
        Self(shape)
    }
}

impl TryCastFrom<Value> for Shape {
    fn can_cast_from(value: &Value) -> bool {
        value.matches::<SmallVec<[u64; 8]>>()
    }

    fn opt_cast_from(value: Value) -> Option<Shape> {
        value.opt_cast_into().map(Self)
    }
}

impl CastFrom<Shape> for Tuple<Value> {
    fn cast_from(shape: Shape) -> Self {
        shape.0.into_iter().collect()
    }
}

impl CastFrom<Shape> for Value {
    fn cast_from(shape: Shape) -> Self {
        Value::Tuple(shape.cast_into())
    }
}

#[async_trait]
impl de::FromStream for Shape {
    type Context = ();

    async fn from_stream<D: de::Decoder>(cxt: (), decoder: &mut D) -> Result<Self, D::Error> {
        de::FromStream::from_stream(cxt, decoder)
            .map_ok(|shape| Self(shape))
            .await
    }
}

impl<'en> en::IntoStream<'en> for Shape {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        self.0.into_stream(encoder)
    }
}

impl<'en> en::ToStream<'en> for Shape {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        self.0.to_stream(encoder)
    }
}

impl fmt::Debug for Shape {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

#[inline]
fn to_range(indices: &[u64]) -> ops::Range<u64> {
    debug_assert!(!indices.is_empty());
    let start = *indices.iter().fold(&u64::MAX, Ord::min);
    let stop = *indices.iter().fold(&0, Ord::max);
    start..stop
}
