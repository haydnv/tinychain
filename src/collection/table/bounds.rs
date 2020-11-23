use std::collections::HashMap;
use std::fmt;

use crate::collection::btree::BTreeRange;
use crate::collection::schema::Column;
use crate::error::{self, TCResult};
use crate::scalar::{
    Bound, Id, Object, Range, Scalar, ScalarClass, ScalarInstance, TryCastFrom, TryCastInto, Value,
    ValueType,
};

#[derive(Clone)]
pub enum ColumnBound {
    Is(Value),
    In(Range),
}

impl From<Value> for ColumnBound {
    fn from(value: Value) -> Self {
        Self::Is(value)
    }
}

impl From<(Bound,Bound)> for ColumnBound {
    fn from(range: (Bound, Bound)) -> Self {
        let (start, end) = range;
        Self::In(Range(start,end))
    }
}

impl TryCastFrom<Scalar> for ColumnBound {
    fn can_cast_from(scalar: &Scalar) -> bool {
        scalar.matches::<Value>() || scalar.matches::<Range>()
    }

    fn opt_cast_from(scalar: Scalar) -> Option<ColumnBound> {
        if scalar.matches::<Range>() {
            scalar.opt_cast_into().map(ColumnBound::In)
        } else {
            scalar.opt_cast_into().map(ColumnBound::Is)
        }
    }
}

impl fmt::Display for ColumnBound {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Is(value) => write!(f, "{}", value),
            Self::In(Range(Bound::Unbounded, Bound::Unbounded)) => write!(f, "[...]"),
            Self::In(Range(start, end)) => {
                match start {
                    Bound::Unbounded => write!(f, "[...")?,
                    Bound::In(value) => write!(f, "[{},", value)?,
                    Bound::Ex(value) => write!(f, "({},", value)?,
                };
                match end {
                    Bound::Unbounded => write!(f, "...]"),
                    Bound::In(value) => write!(f, "{}]", value),
                    Bound::Ex(value) => write!(f, "{})", value),
                }
            }
        }
    }
}

pub type Bounds = HashMap<Id, ColumnBound>;

pub fn all() -> Bounds {
    HashMap::new()
}

pub fn btree_range(bounds: &Bounds, columns: &[Column]) -> TCResult<BTreeRange> {
    let mut start = Vec::with_capacity(bounds.len());
    let mut end = Vec::with_capacity(bounds.len());
    let column_names: Vec<Id> = columns.iter().map(|c| c.name()).cloned().collect();

    use Bound::*;
    for name in &column_names[0..bounds.len()] {
        let bound = bounds
            .get(&name)
            .cloned()
            .ok_or_else(|| error::not_found(name))?;
        match bound {
            ColumnBound::Is(value) => {
                start.push(In(value.clone()));
                end.push(In(value));
            }
            ColumnBound::In(Range(s, e)) => {
                start.push(s);
                end.push(e);
            }
        }
    }

    Ok((start, end).into())
}

pub fn from_key(key: Vec<Value>, key_columns: &[Column]) -> Bounds {
    assert_eq!(key.len(), key_columns.len());

    key_columns
        .iter()
        .map(|c| c.name())
        .cloned()
        .zip(key.into_iter().map(|v| v.into()))
        .collect()
}

impl TryCastFrom<Object> for Bounds {
    fn can_cast_from(object: &Object) -> bool {
        object.values().all(|v| v.matches::<ColumnBound>())
    }

    fn opt_cast_from(object: Object) -> Option<Bounds> {
        let mut bounds = HashMap::new();

        for (id, bound) in object.into_iter() {
            if let Some(bound) = bound.opt_cast_into() {
                bounds.insert(id, bound);
            } else {
                return None;
            }
        }

        Some(bounds)
    }
}

pub fn format(bounds: &Bounds) -> String {
    let bounds: Vec<String> = bounds
        .iter()
        .map(|(k, v)| format!("{}: {}", k, v))
        .collect();

    format!("{{{}}}", bounds.join(", "))
}

pub fn validate(bounds: Bounds, columns: &[Column]) -> TCResult<Bounds> {
    let try_cast_bound = |bound: Bound, dtype: ValueType| match bound {
        Bound::In(val) => dtype.try_cast(val).map(Bound::In),
        Bound::Ex(val) => dtype.try_cast(val).map(Bound::Ex),
        Bound::Unbounded => Ok(Bound::Unbounded),
    };

    let mut validated = Bounds::new();
    let columns: HashMap<&Id, ValueType> = columns.iter().map(|c| c.into()).collect();
    for (name, bound) in bounds.into_iter() {
        if let Some(dtype) = columns.get(&name) {
            let bound = match bound {
                ColumnBound::Is(value) => dtype.try_cast(value).map(ColumnBound::Is)?,
                ColumnBound::In(Range(start, end)) => {
                    let start = try_cast_bound(start, *dtype)?;
                    let end = try_cast_bound(end, *dtype)?;
                    ColumnBound::In(Range(start, end))
                }
            };

            validated.insert(name, bound);
        } else {
            return Err(error::not_found(name));
        }
    }

    Ok(validated)
}
