use std::collections::{HashMap, HashSet};
use std::fmt;
use std::ops::Bound;

use crate::class::{Instance, TCResult};
use crate::collection::btree::BTreeRange;
use crate::collection::schema::Column;
use crate::error;
use crate::scalar::{Value, ValueId, ValueType};

#[derive(Clone)]
pub enum ColumnBound {
    Is(Value),
    In(Bound<Value>, Bound<Value>),
}

impl ColumnBound {
    pub fn expect<M: fmt::Display>(&self, dtype: ValueType, err_context: &M) -> TCResult<()> {
        match self {
            Self::Is(value) => value.expect(dtype, err_context),
            Self::In(start, end) => match start {
                Bound::Included(value) => value.expect(dtype, err_context),
                Bound::Excluded(value) => value.expect(dtype, err_context),
                Bound::Unbounded => Ok(()),
            }
            .and_then(|_| match end {
                Bound::Included(value) => value.expect(dtype, err_context),
                Bound::Excluded(value) => value.expect(dtype, err_context),
                Bound::Unbounded => Ok(()),
            }),
        }
    }
}

impl From<Value> for ColumnBound {
    fn from(value: Value) -> ColumnBound {
        ColumnBound::Is(value)
    }
}

impl fmt::Display for ColumnBound {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Is(value) => write!(f, "{}", value),
            Self::In(Bound::Unbounded, Bound::Unbounded) => write!(f, "[...]"),
            Self::In(start, end) => {
                match start {
                    Bound::Unbounded => write!(f, "[...")?,
                    Bound::Included(value) => write!(f, "[{},", value)?,
                    Bound::Excluded(value) => write!(f, "({},", value)?,
                };
                match end {
                    Bound::Unbounded => write!(f, "...]"),
                    Bound::Included(value) => write!(f, "{}]", value),
                    Bound::Excluded(value) => write!(f, "{})", value),
                }
            }
        }
    }
}

pub type Bounds = HashMap<ValueId, ColumnBound>;

pub fn all() -> Bounds {
    HashMap::new()
}

pub fn btree_range(bounds: &Bounds, columns: &[Column]) -> TCResult<BTreeRange> {
    let mut start = Vec::with_capacity(bounds.len());
    let mut end = Vec::with_capacity(bounds.len());
    let column_names: Vec<ValueId> = columns.iter().map(|c| c.name()).cloned().collect();

    use Bound::*;
    for name in &column_names[0..bounds.len()] {
        let bound = bounds
            .get(&name)
            .cloned()
            .ok_or_else(|| error::not_found(name))?;
        match bound {
            ColumnBound::Is(value) => {
                start.push(Included(value.clone()));
                end.push(Included(value));
            }
            ColumnBound::In(s, e) => {
                start.push(s);
                end.push(e);
            }
        }
    }

    Ok((start, end).into())
}

pub fn from_key(mut key: Vec<Value>, key_columns: &[Column]) -> Bounds {
    assert!(key.len() == key_columns.len());
    key_columns
        .iter()
        .map(|c| c.name())
        .cloned()
        .zip(key.drain(..).map(|v| v.into()))
        .collect()
}

pub fn format(bounds: &Bounds) -> String {
    let bounds: Vec<String> = bounds
        .iter()
        .map(|(k, v)| format!("{}: {}", k, v))
        .collect();

    format!("{{{}}}", bounds.join(", "))
}

pub fn validate(bounds: &Bounds, columns: &[Column]) -> TCResult<()> {
    let column_names: HashSet<&ValueId> = columns.iter().map(|c| c.name()).collect();
    for name in bounds.keys() {
        if !column_names.contains(name) {
            return Err(error::not_found(name));
        }
    }

    Ok(())
}
