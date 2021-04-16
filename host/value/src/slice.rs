use std::fmt;

use async_trait::async_trait;
use collate::Collate;
use destream::{de, en};
use futures::TryFutureExt;
use safecast::{TryCastFrom, TryCastInto};

use tc_error::*;
use tcgeneric::{label, Id, Label, Map};

use super::{Value, ValueCollator};

pub const IN: Label = label("in");
pub const EX: Label = label("ex");

#[derive(Clone, Eq, PartialEq)]
pub enum Bound {
    In(Value),
    Ex(Value),
    Un,
}

impl<'en> en::IntoStream<'en> for Bound {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::In(v) => single_entry(Id::from(IN), v, encoder),
            Self::Ex(v) => single_entry(Id::from(EX), v, encoder),
            Self::Un => Value::None.into_stream(encoder),
        }
    }
}

impl<'en> en::ToStream<'en> for Bound {
    fn to_stream<E: en::Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::In(v) => single_entry(Id::from(IN), v, encoder),
            Self::Ex(v) => single_entry(Id::from(EX), v, encoder),
            Self::Un => Value::None.to_stream(encoder),
        }
    }
}

impl TryCastFrom<Map<Value>> for Bound {
    fn can_cast_from(value: &Map<Value>) -> bool {
        value.len() == 1 && (value.contains_key(&IN.into()) || value.contains_key(&EX.into()))
    }

    fn opt_cast_from(mut value: Map<Value>) -> Option<Self> {
        if value.len() == 1 {
            if let Some(value) = value.remove(&IN.into()) {
                Some(Self::In(value))
            } else if let Some(value) = value.remove(&EX.into()) {
                Some(Self::Ex(value))
            } else {
                None
            }
        } else {
            None
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

#[derive(Clone, Eq, PartialEq)]
pub struct Range {
    pub start: Bound,
    pub end: Bound,
}

impl Range {
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

    pub fn into_inner(self) -> (Bound, Bound) {
        (self.start, self.end)
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

#[async_trait]
impl de::FromStream for Range {
    type Context = ();

    async fn from_stream<D: de::Decoder>(cxt: (), decoder: &mut D) -> Result<Self, D::Error> {
        let start = if let Ok(map) = Map::<Value>::from_stream(cxt, decoder).await {
            map.try_cast_into(|v| TCError::bad_request("invalid Range", v))
                .map_err(de::Error::custom)
        } else {
            Value::from_stream(cxt, decoder)
                .map_ok(|v| if v.is_none() { Bound::Un } else { Bound::In(v) })
                .await
        }?;

        let end = if let Ok(map) = Map::<Value>::from_stream(cxt, decoder).await {
            map.try_cast_into(|v| TCError::bad_request("invalid Range", v))
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

fn single_entry<
    'en,
    K: en::IntoStream<'en> + 'en,
    V: en::IntoStream<'en> + 'en,
    E: en::Encoder<'en>,
>(
    key: K,
    value: V,
    encoder: E,
) -> Result<E::Ok, E::Error> {
    use en::EncodeMap;

    let mut map = encoder.encode_map(Some(1))?;
    map.encode_entry(key, value)?;
    map.end()
}
