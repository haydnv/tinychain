//! A generic map whose keys are [`Id`]s

use std::collections::BTreeMap;
use std::fmt;
use std::iter::FromIterator;
use std::ops::{Deref, DerefMut};

use async_trait::async_trait;
use destream::de::{Decoder, FromStream};
use destream::en::{Encoder, IntoStream, ToStream};
use safecast::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use tc_error::*;

use super::{Id, Tuple};

/// A generic map whose keys are [`Id`]s, based on [`BTreeMap`]
#[derive(Clone)]
pub struct Map<T> {
    inner: BTreeMap<Id, T>,
}

impl<T> Map<T> {
    pub fn new() -> Self {
        Self {
            inner: BTreeMap::new(),
        }
    }

    #[inline]
    pub fn expect_empty(&self) -> TCResult<()>
    where
        T: fmt::Display,
    {
        if self.is_empty() {
            Ok(())
        } else {
            Err(TCError::bad_request("unexpected parameters", self))
        }
    }

    pub fn into_inner(self) -> BTreeMap<Id, T> {
        self.inner
    }

    pub fn option<P, D>(&mut self, name: &Id, default: D) -> TCResult<P>
    where
        P: TryCastFrom<T>,
        D: FnOnce() -> P,
        T: fmt::Display,
    {
        if let Some(param) = self.remove(name) {
            P::try_cast_from(param, |p| {
                TCError::bad_request(format!("invalid value for {}", name), p)
            })
        } else {
            Ok((default)())
        }
    }

    pub fn or_default<P>(&mut self, name: &Id) -> TCResult<P>
    where
        P: Default + TryCastFrom<T>,
        T: fmt::Display,
    {
        if let Some(param) = self.remove(name) {
            P::try_cast_from(param, |p| {
                TCError::bad_request(format!("invalid value for {}", name), p)
            })
        } else {
            Ok(P::default())
        }
    }

    pub fn require<P>(&mut self, name: &Id) -> TCResult<P>
    where
        P: TryCastFrom<T>,
        T: fmt::Display,
    {
        let param = self
            .remove(name)
            .ok_or_else(|| TCError::bad_request("missing required parameter", name))?;

        P::try_cast_from(param, |p| {
            TCError::bad_request(format!("invalid value for {}", name), p)
        })
    }
}

impl<T> Default for Map<T> {
    fn default() -> Map<T> {
        BTreeMap::new().into()
    }
}

impl<T: PartialEq> PartialEq for Map<T> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<T: PartialEq + Eq> Eq for Map<T> {}

impl<T> AsRef<BTreeMap<Id, T>> for Map<T> {
    fn as_ref(&self) -> &BTreeMap<Id, T> {
        &self.inner
    }
}

impl<T> Deref for Map<T> {
    type Target = BTreeMap<Id, T>;

    fn deref(&'_ self) -> &'_ Self::Target {
        &self.inner
    }
}

impl<T> DerefMut for Map<T> {
    fn deref_mut(&'_ mut self) -> &'_ mut <Self as Deref>::Target {
        &mut self.inner
    }
}

impl<T> Extend<(Id, T)> for Map<T> {
    fn extend<I: IntoIterator<Item = (Id, T)>>(&mut self, iter: I) {
        for (key, value) in iter.into_iter() {
            self.insert(key, value);
        }
    }
}

impl<T> IntoIterator for Map<T> {
    type Item = (Id, T);
    type IntoIter = <BTreeMap<Id, T> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

impl<F, T> FromIterator<(Id, F)> for Map<T>
where
    T: CastFrom<F>,
{
    fn from_iter<I: IntoIterator<Item = (Id, F)>>(iter: I) -> Self {
        let mut inner = BTreeMap::new();
        for (id, f) in iter {
            inner.insert(id, f.cast_into());
        }
        Map { inner }
    }
}

impl<T> From<BTreeMap<Id, T>> for Map<T> {
    fn from(inner: BTreeMap<Id, T>) -> Self {
        Map { inner }
    }
}

impl<F, T> TryCastFrom<Tuple<F>> for Map<T>
where
    (Id, T): TryCastFrom<F>,
{
    fn can_cast_from(tuple: &Tuple<F>) -> bool {
        tuple.iter().all(|e| e.matches::<(Id, T)>())
    }

    fn opt_cast_from(tuple: Tuple<F>) -> Option<Self> {
        let mut inner = BTreeMap::<Id, T>::new();

        for f in tuple.into_iter() {
            if let Some((id, t)) = f.opt_cast_into() {
                inner.insert(id, t);
            } else {
                return None;
            }
        }

        Some(Self { inner })
    }
}

#[async_trait]
impl<T: FromStream<Context = ()>> FromStream for Map<T>
where
    T::Context: Copy,
{
    type Context = T::Context;

    async fn from_stream<D: Decoder>(context: T::Context, d: &mut D) -> Result<Self, D::Error> {
        let inner = BTreeMap::<Id, T>::from_stream(context, d).await?;
        Ok(Self { inner })
    }
}

impl<'en, T: IntoStream<'en> + 'en> IntoStream<'en> for Map<T> {
    fn into_stream<E: Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        self.inner.into_stream(encoder)
    }
}

impl<'en, T: ToStream<'en> + 'en> ToStream<'en> for Map<T> {
    fn to_stream<E: Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        self.inner.to_stream(encoder)
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Map<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        BTreeMap::deserialize(deserializer).map(|inner| Self { inner })
    }
}

impl<T: Serialize> Serialize for Map<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.inner.serialize(serializer)
    }
}

impl<F, T: TryCastFrom<F>> TryCastFrom<Map<F>> for BTreeMap<Id, T> {
    fn can_cast_from(map: &Map<F>) -> bool {
        map.values().all(|f| T::can_cast_from(f))
    }

    fn opt_cast_from(source: Map<F>) -> Option<Self> {
        let mut map = BTreeMap::new();

        for (id, f) in source.into_iter() {
            if let Some(t) = T::opt_cast_from(f) {
                map.insert(id, t);
            } else {
                return None;
            }
        }

        Some(map)
    }
}

impl<T: fmt::Debug> fmt::Debug for Map<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_empty() {
            return f.write_str("{}");
        }

        write!(
            f,
            "{{\n{}\n}}",
            self.inner
                .iter()
                .map(|(k, v)| format!("\t{}: {:?}", k, v))
                .collect::<Vec<String>>()
                .join(",\n")
        )
    }
}

impl<T: fmt::Display> fmt::Display for Map<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_empty() {
            return f.write_str("{}");
        }

        write!(
            f,
            "{{ {} }}",
            self.inner
                .iter()
                .map(|(k, v)| format!("{}: {}", k, v))
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}
