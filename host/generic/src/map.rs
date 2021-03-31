//! A generic map whose keys are [`Id`]s

use std::collections::HashMap;
use std::fmt;
use std::iter::FromIterator;
use std::ops::{Deref, DerefMut};

use async_trait::async_trait;
use destream::de::{Decoder, FromStream};
use destream::en::{Encoder, IntoStream, ToStream};
use safecast::{Match, TryCastFrom, TryCastInto};

use tc_error::*;

use super::{Id, Tuple};

/// A generic map whose keys are [`Id`]s, based on [`HashMap`]
#[derive(Clone)]
pub struct Map<T: Clone> {
    inner: HashMap<Id, T>,
}

impl<T: Clone> Map<T> {
    #[inline]
    pub fn expect_empty(&self) -> TCResult<()> where T: fmt::Display {
        if self.is_empty() {
            Ok(())
        } else {
            Err(TCError::bad_request("unexpected parameters", self))
        }
    }

    pub fn into_inner(self) -> HashMap<Id, T> {
        self.inner
    }

    pub fn or_default<P: Default + TryCastFrom<T>>(&mut self, name: &Id) -> TCResult<P> where T: fmt::Display {
        if let Some(param) = self.remove(name) {
            P::try_cast_from(param, |p| {
                TCError::bad_request(format!("invalid value for {}", name), p)
            })
        } else {
            Ok(P::default())
        }
    }

    pub fn require<P: TryCastFrom<T>>(&mut self, name: &Id) -> TCResult<P> where T: fmt::Display {
        let param = self.remove(name).ok_or_else(|| TCError::bad_request("missing required parameter", name))?;
        P::try_cast_from(param, |p| TCError::bad_request(format!("invalid value for {}", name), p))
    }
}

impl<T: Clone> Default for Map<T> {
    fn default() -> Map<T> {
        HashMap::new().into()
    }
}

impl<T: Clone + PartialEq> PartialEq for Map<T> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<T: Clone + PartialEq + Eq> Eq for Map<T> {}

impl<T: Clone> AsRef<HashMap<Id, T>> for Map<T> {
    fn as_ref(&self) -> &HashMap<Id, T> {
        &self.inner
    }
}

impl<T: Clone> Deref for Map<T> {
    type Target = HashMap<Id, T>;

    fn deref(&'_ self) -> &'_ Self::Target {
        &self.inner
    }
}

impl<T: Clone> DerefMut for Map<T> {
    fn deref_mut(&'_ mut self) -> &'_ mut <Self as Deref>::Target {
        &mut self.inner
    }
}

impl<T: Clone> IntoIterator for Map<T> {
    type Item = (Id, T);
    type IntoIter = <HashMap<Id, T> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

impl<T: Clone> FromIterator<(Id, T)> for Map<T> {
    fn from_iter<I: IntoIterator<Item = (Id, T)>>(iter: I) -> Self {
        let inner = HashMap::from_iter(iter);
        Map { inner }
    }
}

impl<T: Clone> From<HashMap<Id, T>> for Map<T> {
    fn from(inner: HashMap<Id, T>) -> Self {
        Map { inner }
    }
}

impl<F: Clone, T: Clone> TryCastFrom<Tuple<F>> for Map<T>
where
    (Id, T): TryCastFrom<F>,
{
    fn can_cast_from(tuple: &Tuple<F>) -> bool {
        tuple.iter().all(|e| e.matches::<(Id, T)>())
    }

    fn opt_cast_from(tuple: Tuple<F>) -> Option<Self> {
        let mut inner = HashMap::<Id, T>::new();

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
impl<T: Clone + FromStream<Context = ()>> FromStream for Map<T>
where
    T::Context: Copy,
{
    type Context = T::Context;

    async fn from_stream<D: Decoder>(context: T::Context, d: &mut D) -> Result<Self, D::Error> {
        let inner = HashMap::<Id, T>::from_stream(context, d).await?;
        Ok(Self { inner })
    }
}

impl<'en, T: Clone + IntoStream<'en> + 'en> IntoStream<'en> for Map<T> {
    fn into_stream<E: Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        self.inner.into_stream(encoder)
    }
}

impl<'en, T: Clone + ToStream<'en> + 'en> ToStream<'en> for Map<T> {
    fn to_stream<E: Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        self.inner.to_stream(encoder)
    }
}

impl<F: Clone, T: TryCastFrom<F>> TryCastFrom<Map<F>> for HashMap<Id, T> {
    fn can_cast_from(map: &Map<F>) -> bool {
        map.values().all(|f| T::can_cast_from(f))
    }

    fn opt_cast_from(source: Map<F>) -> Option<Self> {
        let mut map = HashMap::new();

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

impl<T: Clone + fmt::Display> fmt::Display for Map<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_empty() {
            return f.write_str("{}");
        }

        write!(
            f,
            "{{\n{}\n}}",
            self.inner
                .iter()
                .map(|(k, v)| format!("\t{}: {}", k, v))
                .collect::<Vec<String>>()
                .join(",\n")
        )
    }
}
