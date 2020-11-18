use std::cmp::Ordering;
use std::fmt;

use crate::collection::schema::Column;
use crate::error::TCResult;
use crate::scalar::*;

use super::collator;
use super::Key;

#[derive(Clone, Eq, PartialEq)]
pub struct BTreeRange(Vec<Bound>, Vec<Bound>);

impl BTreeRange {
    pub fn contains(&self, other: &BTreeRange, schema: &[Column]) -> TCResult<bool> {
        if other == &Self::default() {
            return Ok(true);
        }

        if other.0.len() < self.0.len() {
            return Ok(false);
        }

        if other.1.len() < self.1.len() {
            return Ok(false);
        }

        use collator::compare_value;
        use Bound::*;
        use Ordering::*;

        for (col, (outer, inner)) in schema[0..self.0.len()]
            .iter()
            .zip(self.0.iter().zip(other.0[0..self.0.len()].iter()))
        {
            let dtype = *col.dtype();

            match (outer, inner) {
                (Unbounded, _) => {}
                (_, Unbounded) => return Ok(false),
                (Ex(o), Ex(i)) if compare_value(&o, &i, dtype)? == Greater => return Ok(false),
                (In(o), In(i)) if compare_value(&o, &i, dtype)? == Greater => return Ok(false),
                (In(o), Ex(i)) if compare_value(&o, &i, dtype)? == Greater => return Ok(false),
                (Ex(o), In(i)) if compare_value(&o, &i, dtype)? != Less => return Ok(false),
                _ => {}
            }
        }

        for (col, (outer, inner)) in schema[0..self.1.len()]
            .iter()
            .zip(self.1.iter().zip(other.1[0..self.1.len()].iter()))
        {
            let dtype = *col.dtype();

            match (outer, inner) {
                (Unbounded, _) => {}
                (_, Unbounded) => return Ok(false),
                (Ex(o), Ex(i)) if compare_value(&o, &i, dtype)? == Less => return Ok(false),
                (In(o), In(i)) if compare_value(&o, &i, dtype)? == Less => return Ok(false),
                (In(o), Ex(i)) if compare_value(&o, &i, dtype)? == Less => return Ok(false),
                (Ex(o), In(i)) if compare_value(&o, &i, dtype)? != Greater => return Ok(false),
                _ => {}
            }
        }

        Ok(true)
    }

    pub fn is_key(&self, schema: &[Column]) -> bool {
        self.0.len() == self.1.len()
            && self.0.len() == schema.len()
            && self.0.iter().zip(self.1.iter()).all(|(l, r)| l == r)
    }

    pub fn start(&'_ self) -> &'_ [Bound] {
        &self.0
    }

    pub fn end(&'_ self) -> &'_ [Bound] {
        &self.1
    }
}

pub fn validate_range(range: BTreeRange, schema: &[Column]) -> TCResult<BTreeRange> {
    use Bound::*;

    let cast = |(bound, column): (Bound, &Column)| {
        let value = match bound {
            Unbounded => Unbounded,
            In(value) => In(column.dtype().try_cast(value)?),
            Ex(value) => Ex(column.dtype().try_cast(value)?),
        };
        Ok(value)
    };

    let cast_range = |range: Vec<Bound>| {
        range
            .into_iter()
            .zip(schema)
            .map(cast)
            .collect::<TCResult<Vec<Bound>>>()
    };

    let start = cast_range(range.0)?;
    let end = cast_range(range.1)?;
    Ok(BTreeRange(start, end))
}

impl Default for BTreeRange {
    fn default() -> Self {
        Self(vec![], vec![])
    }
}

impl From<Key> for BTreeRange {
    fn from(mut key: Key) -> Self {
        let start = key.iter().cloned().map(Bound::In).collect();
        let end = key.drain(..).map(Bound::In).collect();
        Self(start, end)
    }
}

impl From<(Vec<Bound>, Vec<Bound>)> for BTreeRange {
    fn from(params: (Vec<Bound>, Vec<Bound>)) -> Self {
        Self(params.0, params.1)
    }
}

impl From<Vec<Range>> for BTreeRange {
    fn from(range: Vec<Range>) -> Self {
        Self::from(range.into_iter().map(Range::into_inner).unzip())
    }
}

impl TryCastFrom<Value> for BTreeRange {
    fn can_cast_from(value: &Value) -> bool {
        if value == &Value::None || Key::can_cast_from(value) {
            true
        } else if let Value::Tuple(tuple) = value {
            tuple.iter().all(|v| v.is_none())
        } else {
            false
        }
    }

    fn opt_cast_from(value: Value) -> Option<BTreeRange> {
        if value == Value::None {
            Some(BTreeRange::default())
        } else if let Value::Tuple(tuple) = value {
            if tuple.iter().all(|v| v.is_none()) {
                Some(BTreeRange::default())
            } else {
                None
            }
        } else {
            Key::opt_cast_from(value).map(BTreeRange::from)
        }
    }
}

impl TryCastFrom<Scalar> for BTreeRange {
    fn can_cast_from(scalar: &Scalar) -> bool {
        match scalar {
            Scalar::Value(value) => Self::can_cast_from(value),
            range if range.matches::<Vec<Range>>() => true,
            Scalar::Tuple(tuple) => Value::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(scalar: Scalar) -> Option<BTreeRange> {
        match scalar {
            Scalar::Value(value) => Self::opt_cast_from(value),
            range if range.matches::<Vec<Range>>() => {
                let range: Vec<Range> = range.opt_cast_into().unwrap();
                Some(Self::from(range))
            }
            Scalar::Tuple(tuple) if Value::can_cast_from(&tuple) => {
                let value = Value::opt_cast_from(tuple).unwrap();
                Self::opt_cast_from(value)
            }
            _ => None,
        }
    }
}

impl fmt::Display for BTreeRange {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.0.is_empty() && self.1.is_empty() {
            return write!(f, "BTreeRange::default");
        }

        let to_str = |bounds: &[Bound]| {
            bounds
                .iter()
                .map(|bound| bound.to_string())
                .collect::<Vec<String>>()
                .join(", ")
        };

        write!(f, "BTreeRange: ({}, {})", to_str(&self.0), to_str(&self.1))
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct Selector {
    range: BTreeRange,
    reverse: bool,
}

impl Selector {
    pub fn range(&'_ self) -> &'_ BTreeRange {
        &self.range
    }

    pub fn reverse(&'_ self) -> bool {
        self.reverse
    }

    pub fn into_inner(self) -> (BTreeRange, bool) {
        (self.range, self.reverse)
    }
}

impl Default for Selector {
    fn default() -> Selector {
        Selector {
            range: BTreeRange::default(),
            reverse: false,
        }
    }
}

impl From<BTreeRange> for Selector {
    fn from(range: BTreeRange) -> Selector {
        Selector {
            range,
            reverse: false,
        }
    }
}

impl From<(BTreeRange, bool)> for Selector {
    fn from(range: (BTreeRange, bool)) -> Selector {
        let (range, reverse) = range;
        Selector { range, reverse }
    }
}

impl fmt::Display for Selector {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "BTree selector with range {} (reverse: {})",
            self.range, self.reverse
        )
    }
}
