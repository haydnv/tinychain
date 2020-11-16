use std::cmp::Ordering;
use std::ops::Bound;

use crate::collection::schema::Column;
use crate::error::TCResult;
use crate::scalar::{CastFrom, ScalarClass, TryCastFrom, Value, ValueType};

use super::collator;
use super::Key;

#[derive(Clone, Eq, PartialEq)]
pub struct BTreeRange(Vec<Bound<Value>>, Vec<Bound<Value>>);

impl BTreeRange {
    pub fn contains(&self, other: &BTreeRange, schema: &[ValueType]) -> TCResult<bool> {
        if other.0.len() < self.0.len() {
            return Ok(false);
        }

        if other.1.len() < self.1.len() {
            return Ok(false);
        }

        use collator::compare_value;
        use Bound::*;
        use Ordering::*;

        for (dtype, (outer, inner)) in schema[0..self.0.len()]
            .iter()
            .zip(self.0.iter().zip(other.0[0..self.0.len()].iter()))
        {
            match (outer, inner) {
                (Unbounded, _) => {}
                (_, Unbounded) => return Ok(false),
                (Excluded(o), Excluded(i)) if compare_value(&o, &i, *dtype)? == Greater => {
                    return Ok(false)
                }
                (Included(o), Included(i)) if compare_value(&o, &i, *dtype)? == Greater => {
                    return Ok(false)
                }
                (Included(o), Excluded(i)) if compare_value(&o, &i, *dtype)? == Greater => {
                    return Ok(false)
                }
                (Excluded(o), Included(i)) if compare_value(&o, &i, *dtype)? != Less => {
                    return Ok(false)
                }
                _ => {}
            }
        }

        for (dtype, (outer, inner)) in schema[0..self.1.len()]
            .iter()
            .zip(self.1.iter().zip(other.1[0..self.1.len()].iter()))
        {
            match (outer, inner) {
                (Unbounded, _) => {}
                (_, Unbounded) => return Ok(false),
                (Excluded(o), Excluded(i)) if compare_value(&o, &i, *dtype)? == Less => {
                    return Ok(false)
                }
                (Included(o), Included(i)) if compare_value(&o, &i, *dtype)? == Less => {
                    return Ok(false)
                }
                (Included(o), Excluded(i)) if compare_value(&o, &i, *dtype)? == Less => {
                    return Ok(false)
                }
                (Excluded(o), Included(i)) if compare_value(&o, &i, *dtype)? != Greater => {
                    return Ok(false)
                }
                _ => {}
            }
        }

        Ok(true)
    }

    pub fn start(&'_ self) -> &'_ [Bound<Value>] {
        &self.0
    }

    pub fn end(&'_ self) -> &'_ [Bound<Value>] {
        &self.1
    }
}

pub fn validate_range(range: BTreeRange, schema: &[Column]) -> TCResult<BTreeRange> {
    use Bound::*;

    let cast = |(bound, column): (Bound<Value>, &Column)| {
        let value = match bound {
            Unbounded => Unbounded,
            Included(value) => Included(column.dtype().try_cast(value)?),
            Excluded(value) => Excluded(column.dtype().try_cast(value)?),
        };
        Ok(value)
    };

    let cast_range = |range: Vec<Bound<Value>>| {
        range
            .into_iter()
            .zip(schema)
            .map(cast)
            .collect::<TCResult<Vec<Bound<Value>>>>()
    };

    let start = cast_range(range.0)?;
    let end = cast_range(range.1)?;
    Ok(BTreeRange(start, end))
}

impl Default for BTreeRange {
    fn default() -> BTreeRange {
        BTreeRange(vec![], vec![])
    }
}

impl From<Key> for BTreeRange {
    fn from(mut key: Key) -> BTreeRange {
        let start = key.iter().cloned().map(Bound::Included).collect();
        let end = key.drain(..).map(Bound::Included).collect();
        BTreeRange(start, end)
    }
}

impl From<(Vec<Bound<Value>>, Vec<Bound<Value>>)> for BTreeRange {
    fn from(params: (Vec<Bound<Value>>, Vec<Bound<Value>>)) -> BTreeRange {
        BTreeRange(params.0, params.1)
    }
}

impl TryCastFrom<Value> for BTreeRange {
    fn can_cast_from(value: &Value) -> bool {
        value == &Value::None || Key::can_cast_from(value)
    }

    fn opt_cast_from(value: Value) -> Option<BTreeRange> {
        if value == Value::None {
            Some(BTreeRange::default())
        } else {
            Key::opt_cast_from(value).map(BTreeRange::from)
        }
    }
}

impl CastFrom<BTreeRange> for Value {
    fn cast_from(_s: BTreeRange) -> Value {
        unimplemented!()
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
