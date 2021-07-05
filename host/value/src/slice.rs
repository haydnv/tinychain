use std::fmt;
use std::ops;

use async_trait::async_trait;
use collate::Collate;
use destream::{de, en};
use futures::TryFutureExt;
use safecast::{Match, TryCastFrom, TryCastInto};

use tc_error::*;
use tcgeneric::{label, Id, Label, Tuple};

use super::{Value, ValueCollator};

/// The prefix of an inclusive [`Bound`]
pub const IN: Label = label("in");

/// The prefix of an exclusive [`Bound`]
pub const EX: Label = label("ex");

/// An optional inclusive or exclusive bound
#[derive(Clone, Eq, PartialEq)]
pub enum Bound {
    In(Value),
    Ex(Value),
    Un,
}

impl<'en> en::IntoStream<'en> for Bound {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::In(v) => (Id::from(IN), v).into_stream(encoder),
            Self::Ex(v) => (Id::from(EX), v).into_stream(encoder),
            Self::Un => Value::None.into_stream(encoder),
        }
    }
}

impl<'en> en::ToStream<'en> for Bound {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::In(v) => en::IntoStream::into_stream((Id::from(IN), v), encoder),
            Self::Ex(v) => en::IntoStream::into_stream((Id::from(EX), v), encoder),
            Self::Un => Value::None.to_stream(encoder),
        }
    }
}

impl TryCastFrom<Value> for Bound {
    fn can_cast_from(value: &Value) -> bool {
        match value {
            Value::Tuple(tuple) => Self::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(value: Value) -> Option<Self> {
        match value {
            Value::Tuple(tuple) => tuple.opt_cast_into(),
            _ => None,
        }
    }
}

impl TryCastFrom<Tuple<Value>> for Bound {
    fn can_cast_from(value: &Tuple<Value>) -> bool {
        TryCastInto::<(Id, Value)>::can_cast_into(value)
    }

    fn opt_cast_from(value: Tuple<Value>) -> Option<Self> {
        let (rtype, value): (Id, Value) = value.opt_cast_into()?;

        if rtype == IN {
            Some(Self::In(value))
        } else if rtype == EX {
            Some(Self::Ex(value))
        } else {
            None
        }
    }
}

impl From<Bound> for ops::Bound<Value> {
    fn from(bound: Bound) -> Self {
        match bound {
            Bound::In(value) => Self::Included(value),
            Bound::Ex(value) => Self::Excluded(value),
            Bound::Un => Self::Unbounded,
        }
    }
}

impl fmt::Debug for Bound {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::In(value) => write!(f, "include({:?})", value),
            Self::Ex(value) => write!(f, "exclude({:?})", value),
            Self::Un => write!(f, "unbounded"),
        }
    }
}

impl fmt::Display for Bound {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::In(value) => write!(f, "include({})", value),
            Self::Ex(value) => write!(f, "exclude({})", value),
            Self::Un => write!(f, "unbounded"),
        }
    }
}

/// A range comprising a start and end [`Bound`]
#[derive(Clone, Eq, PartialEq)]
pub struct Range {
    pub start: Bound,
    pub end: Bound,
}

impl Range {
    /// Return true if the given `Range` is within this `Range`.
    pub fn contains_range(&self, inner: &Self, collator: &ValueCollator) -> bool {
        use std::cmp::Ordering::*;

        match &self.start {
            Bound::Un => {}
            Bound::In(outer) => match &inner.start {
                Bound::Un => return false,
                Bound::In(inner) => {
                    if collator.compare(inner, outer) == Less {
                        return false;
                    }
                }
                Bound::Ex(inner) => {
                    if collator.compare(inner, outer) != Greater {
                        return false;
                    }
                }
            },
            Bound::Ex(outer) => match &inner.start {
                Bound::Un => return false,
                Bound::In(inner) => {
                    if collator.compare(inner, outer) != Greater {
                        return false;
                    }
                }
                Bound::Ex(inner) => {
                    if collator.compare(inner, outer) == Less {
                        return false;
                    }
                }
            },
        }

        match &self.end {
            Bound::Un => {}
            Bound::In(outer) => match &inner.end {
                Bound::Un => return false,
                Bound::In(inner) => {
                    if collator.compare(inner, outer) == Greater {
                        return false;
                    }
                }
                Bound::Ex(inner) => {
                    if collator.compare(inner, outer) != Less {
                        return false;
                    }
                }
            },
            Bound::Ex(outer) => match &inner.end {
                Bound::Un => return false,
                Bound::In(inner) => {
                    if collator.compare(inner, outer) != Less {
                        return false;
                    }
                }
                Bound::Ex(inner) => {
                    if collator.compare(inner, outer) == Greater {
                        return false;
                    }
                }
            },
        }

        true
    }

    /// Return true if the given [`Value`] is within this `Range`.
    pub fn contains_value(&self, value: &Value, collator: &ValueCollator) -> bool {
        use std::cmp::Ordering::*;

        match &self.start {
            Bound::Un => {}
            Bound::In(outer) => {
                if collator.compare(value, outer) == Less {
                    return false;
                }
            }
            Bound::Ex(outer) => {
                if collator.compare(value, outer) != Greater {
                    return false;
                }
            }
        }

        match &self.end {
            Bound::Un => {}
            Bound::In(outer) => {
                if collator.compare(value, outer) == Greater {
                    return false;
                }
            }
            Bound::Ex(outer) => {
                if collator.compare(value, outer) != Less {
                    return false;
                }
            }
        }

        true
    }
}

impl Default for Range {
    fn default() -> Self {
        Self {
            start: Bound::Un,
            end: Bound::Un,
        }
    }
}

impl TryCastFrom<Value> for Range {
    fn can_cast_from(value: &Value) -> bool {
        match value {
            Value::Tuple(tuple) => Self::can_cast_from(tuple),
            _ => false,
        }
    }

    fn opt_cast_from(value: Value) -> Option<Self> {
        match value {
            Value::Tuple(tuple) => tuple.opt_cast_into(),
            _ => None,
        }
    }
}

impl TryCastFrom<Tuple<Value>> for Range {
    fn can_cast_from(tuple: &Tuple<Value>) -> bool {
        TryCastInto::<(Value, Value)>::can_cast_into(tuple)
    }

    fn opt_cast_from(tuple: Tuple<Value>) -> Option<Self> {
        let (start, end): (Value, Value) = tuple.opt_cast_into()?;

        let start = if start.matches::<Bound>() {
            match start.opt_cast_into().unwrap() {
                Bound::In(value) if value.is_none() => Bound::Un,
                bound => bound,
            }
        } else {
            Bound::In(start)
        };

        let end = if end.matches::<Bound>() {
            match end.opt_cast_into().unwrap() {
                Bound::Ex(value) if value.is_none() => Bound::Un,
                bound => bound,
            }
        } else {
            Bound::Ex(end)
        };

        Some(Range { start, end })
    }
}

#[async_trait]
impl de::FromStream for Range {
    type Context = ();

    async fn from_stream<D: de::Decoder>(cxt: (), decoder: &mut D) -> Result<Self, D::Error> {
        let start = if let Ok(tuple) = Tuple::<Value>::from_stream(cxt, decoder).await {
            tuple
                .try_cast_into(|v| TCError::bad_request("invalid Range", v))
                .map_err(de::Error::custom)
        } else {
            Value::from_stream(cxt, decoder)
                .map_ok(|v| if v.is_none() { Bound::Un } else { Bound::In(v) })
                .await
        }?;

        let end = if let Ok(tuple) = Tuple::<Value>::from_stream(cxt, decoder).await {
            tuple
                .try_cast_into(|v| TCError::bad_request("invalid Range", v))
                .map_err(de::Error::custom)
        } else {
            Value::from_stream(cxt, decoder)
                .map_ok(|v| if v.is_none() { Bound::Un } else { Bound::Ex(v) })
                .await
        }?;

        Ok(Range { start, end })
    }
}

impl<'en> en::IntoStream<'en> for Range {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        match (self.start, self.end) {
            (Bound::In(start), Bound::Ex(end)) => (start, end).into_stream(encoder),
            (Bound::Un, Bound::Ex(end)) => ((), end).into_stream(encoder),
            (Bound::In(start), Bound::Un) => (start, ()).into_stream(encoder),
            (start, end) => (start, end).into_stream(encoder),
        }
    }
}

impl<'en> en::ToStream<'en> for Range {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        use en::IntoStream;

        match (&self.start, &self.end) {
            (Bound::In(start), Bound::Ex(end)) => (start, end).into_stream(encoder),
            (Bound::Un, Bound::Ex(end)) => ((), end).into_stream(encoder),
            (Bound::In(start), Bound::Un) => (start, ()).into_stream(encoder),
            (start, end) => (start, end).into_stream(encoder),
        }
    }
}

impl From<Range> for (ops::Bound<Value>, ops::Bound<Value>) {
    fn from(range: Range) -> Self {
        (range.start.into(), range.end.into())
    }
}

impl fmt::Debug for Range {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match (&self.start, &self.end) {
            (Bound::Ex(start), Bound::Ex(end)) => write!(f, "({:?}, {:?})", start, end),
            (Bound::Ex(start), Bound::In(end)) => write!(f, "({:?}, {:?}]", start, end),
            (Bound::Ex(start), Bound::Un) => write!(f, "({:?}...]", start),

            (Bound::In(start), Bound::Ex(end)) => write!(f, "[{:?}, {:?})", start, end),
            (Bound::In(start), Bound::In(end)) => write!(f, "[{:?}, {:?}]", start, end),
            (Bound::In(start), Bound::Un) => write!(f, "[{:?}...]", start),

            (Bound::Un, Bound::Ex(end)) => write!(f, "[...{:?})", end),
            (Bound::Un, Bound::In(end)) => write!(f, "[...{:?}]", end),
            (Bound::Un, Bound::Un) => f.write_str("[...]"),
        }
    }
}

impl fmt::Display for Range {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match (&self.start, &self.end) {
            (Bound::Ex(start), Bound::Ex(end)) => write!(f, "({}, {})", start, end),
            (Bound::Ex(start), Bound::In(end)) => write!(f, "({}, {}]", start, end),
            (Bound::Ex(start), Bound::Un) => write!(f, "({}...]", start),

            (Bound::In(start), Bound::Ex(end)) => write!(f, "[{}, {})", start, end),
            (Bound::In(start), Bound::In(end)) => write!(f, "[{}, {}]", start, end),
            (Bound::In(start), Bound::Un) => write!(f, "[{}...]", start),

            (Bound::Un, Bound::Ex(end)) => write!(f, "[...{})", end),
            (Bound::Un, Bound::In(end)) => write!(f, "[...{}]", end),
            (Bound::Un, Bound::Un) => f.write_str("[...]"),
        }
    }
}
