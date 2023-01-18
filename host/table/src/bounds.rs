use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::iter::FromIterator;
use std::ops::{Deref, DerefMut};

use collate::Collate;

use tc_error::*;
use tc_value::{Bound, Range, Value, ValueCollator, ValueType};
use tcgeneric::{Id, Map, Tuple};

use super::Column;

/// A bound on a single [`Column`] of a `Table`.
#[derive(Clone)]
pub enum ColumnBound {
    Is(Value),
    In(Range),
}

impl ColumnBound {
    /// Return true if the given [`ColumnBound`] falls within this one,
    /// according to the given [`ValueCollator`].
    fn contains(&self, inner: &Self, collator: &ValueCollator) -> bool {
        use Ordering::*;

        match self {
            Self::Is(outer) => match inner {
                Self::Is(inner) => collator.compare(outer, inner) == Equal,
                Self::In(Range {
                    start: Bound::In(start),
                    end: Bound::In(end),
                }) => {
                    collator.compare(outer, start) == Equal && collator.compare(outer, end) == Equal
                }
                _ => false,
            },
            Self::In(outer) => match inner {
                Self::Is(inner) => outer.contains_value(inner, collator),
                Self::In(inner) => outer.contains_range(inner, collator),
            },
        }
    }

    /// Return false if this `ColumnBound` is a single [`Value`].
    pub fn is_range(&self) -> bool {
        match self {
            ColumnBound::In(_) => true,
            _ => false,
        }
    }
}

impl Default for ColumnBound {
    fn default() -> Self {
        Self::In(Range::default())
    }
}

impl From<Value> for ColumnBound {
    fn from(value: Value) -> Self {
        Self::Is(value)
    }
}

impl From<(Bound, Bound)> for ColumnBound {
    fn from(range: (Bound, Bound)) -> Self {
        let (start, end) = range;
        Self::In(Range { start, end })
    }
}

impl fmt::Display for ColumnBound {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Is(value) => write!(f, "{}", value),
            Self::In(Range {
                start: Bound::Un,
                end: Bound::Un,
            }) => write!(f, "[...]"),
            Self::In(Range { start, end }) => {
                match start {
                    Bound::Un => write!(f, "[...")?,
                    Bound::In(value) => write!(f, "[{},", value)?,
                    Bound::Ex(value) => write!(f, "({},", value)?,
                };
                match end {
                    Bound::Un => write!(f, "...]"),
                    Bound::In(value) => write!(f, "{}]", value),
                    Bound::Ex(value) => write!(f, "{})", value),
                }
            }
        }
    }
}

/// Selection bounds for a `Table`
#[derive(Clone, Default)]
pub struct Bounds {
    inner: HashMap<Id, ColumnBound>,
}

impl Bounds {
    /// Construct a new `Table` `Bounds` from a given `key` according to the given schema.
    pub fn from_key(key: Vec<Value>, key_columns: &[Column]) -> Self {
        assert_eq!(key.len(), key_columns.len());

        let inner = key_columns
            .iter()
            .map(|c| c.name())
            .cloned()
            .zip(key.into_iter().map(|v| v.into()))
            .collect();

        Self { inner }
    }

    /// Convert these `Bounds` into an equivalent [`tc_btree::Range`] according to the given schema.
    pub fn into_btree_range(mut self, columns: &[Column]) -> TCResult<tc_btree::Range> {
        let on_err = |bounds: &HashMap<Id, ColumnBound>| {
            bad_request!(
                "extra columns in Table bounds {}",
                bounds.keys().collect::<Tuple<&Id>>()
            )
        };

        let mut prefix = Vec::with_capacity(self.len());

        let mut i = 0;
        let range = loop {
            let column = columns.get(i).ok_or_else(|| on_err(&self.inner))?;

            match self.remove(column.name()) {
                None => break tc_btree::Range::with_prefix(prefix),
                Some(ColumnBound::In(Range { start, end })) => {
                    break (prefix, start.into(), end.into()).into()
                }
                Some(ColumnBound::Is(value)) => prefix.push(value),
            }

            i += 1;
        };

        if self.is_empty() {
            Ok(range)
        } else {
            Err(on_err(&self.inner))
        }
    }

    /// Merge these `Bounds` with the given `other`.
    pub fn merge(&mut self, other: Self, collator: &ValueCollator) -> TCResult<()> {
        for (col_name, inner) in other.inner.into_iter() {
            if let Some(outer) = self.get(&col_name) {
                if !outer.contains(&inner, collator) {
                    return Err(bad_request!(
                        "table bounds {} does not contain {} to merge",
                        outer,
                        inner
                    ));
                }
            }

            self.inner.insert(col_name, inner);
        }

        Ok(())
    }

    /// Cast these `Bounds` to match the given schema, or return an error.
    pub fn validate(self, columns: &[Column]) -> TCResult<Bounds> {
        let try_cast_bound = |bound: Bound, dtype: ValueType| match bound {
            Bound::In(val) => dtype.try_cast(val).map(Bound::In),
            Bound::Ex(val) => dtype.try_cast(val).map(Bound::Ex),
            Bound::Un => Ok(Bound::Un),
        };

        let mut validated = HashMap::new();
        let columns: HashMap<&Id, ValueType> = columns.iter().map(|c| c.into()).collect();
        for (name, bound) in self.inner.into_iter() {
            if let Some(dtype) = columns.get(&name) {
                let bound = match bound {
                    ColumnBound::Is(value) => dtype.try_cast(value).map(ColumnBound::Is)?,
                    ColumnBound::In(Range { start, end }) => {
                        let start = try_cast_bound(start, *dtype)?;
                        let end = try_cast_bound(end, *dtype)?;
                        ColumnBound::In(Range { start, end })
                    }
                };

                validated.insert(name, bound);
            } else {
                return Err(TCError::not_found(name));
            }
        }

        Ok(validated.into())
    }
}

impl Deref for Bounds {
    type Target = HashMap<Id, ColumnBound>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for Bounds {
    fn deref_mut(&'_ mut self) -> &'_ mut Self::Target {
        &mut self.inner
    }
}

impl From<HashMap<Id, ColumnBound>> for Bounds {
    fn from(inner: HashMap<Id, ColumnBound>) -> Self {
        Self { inner }
    }
}

impl<V: Into<ColumnBound>> FromIterator<(Id, V)> for Bounds {
    fn from_iter<I: IntoIterator<Item = (Id, V)>>(iter: I) -> Self {
        Self {
            inner: iter
                .into_iter()
                .map(|(id, bound)| (id, bound.into()))
                .collect(),
        }
    }
}

impl fmt::Display for Bounds {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&Map::<ColumnBound>::from_iter(self.inner.clone()), f)
    }
}
