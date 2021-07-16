//! A generic tuple type
//!
//! Note that, from the perspective of a Tinychain user (say, a developer using the HTTP API via
//! the Python client), this really is a tuple, in the sense that it is immutable, with a fixed
//! length, and not iterable.

use std::fmt;
use std::iter::FromIterator;
use std::ops::{Deref, DerefMut};

use async_trait::async_trait;
use destream::de::{Decoder, FromStream};
use destream::en::{Encoder, IntoStream, ToStream};
use safecast::*;

/// A generic tuple type, based on [`Vec`]
#[derive(Clone, Default, Eq, PartialEq)]
pub struct Tuple<T> {
    inner: Vec<T>,
}

impl<T> Tuple<T> {
    pub fn into_inner(self) -> Vec<T> {
        self.inner
    }
}

impl<T> AsRef<Vec<T>> for Tuple<T> {
    fn as_ref(&self) -> &Vec<T> {
        &self.inner
    }
}

impl<T> Deref for Tuple<T> {
    type Target = Vec<T>;

    fn deref(&'_ self) -> &'_ Self::Target {
        &self.inner
    }
}

impl<T> DerefMut for Tuple<T> {
    fn deref_mut(&'_ mut self) -> &'_ mut <Self as Deref>::Target {
        &mut self.inner
    }
}

impl<T: CastFrom<F>, F> FromIterator<F> for Tuple<T> {
    fn from_iter<I: IntoIterator<Item = F>>(iter: I) -> Self {
        let inner = Vec::from_iter(iter.into_iter().map(|f| f.cast_into()));
        Self { inner }
    }
}

impl<T> From<Vec<T>> for Tuple<T> {
    fn from(inner: Vec<T>) -> Self {
        Self { inner }
    }
}

impl<T> From<(T,)> for Tuple<T> {
    fn from(inner: (T,)) -> Self {
        let (a,) = inner;
        Self { inner: vec![a] }
    }
}

impl<T> From<(T, T)> for Tuple<T> {
    fn from(inner: (T, T)) -> Self {
        let (a, b) = inner;
        Self { inner: vec![a, b] }
    }
}

impl<T> From<(T, T, T)> for Tuple<T> {
    fn from(inner: (T, T, T)) -> Self {
        let (a, b, c) = inner;
        Self {
            inner: vec![a, b, c],
        }
    }
}

impl<T> From<(T, T, T, T)> for Tuple<T> {
    fn from(inner: (T, T, T, T)) -> Self {
        let (a, b, c, d) = inner;
        Self {
            inner: vec![a, b, c, d],
        }
    }
}

impl<T> IntoIterator for Tuple<T> {
    type Item = T;
    type IntoIter = <Vec<T> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

impl<F, T: TryCastFrom<F>> TryCastFrom<Tuple<F>> for Vec<T> {
    fn can_cast_from(tuple: &Tuple<F>) -> bool {
        tuple.iter().all(T::can_cast_from)
    }

    fn opt_cast_from(tuple: Tuple<F>) -> Option<Self> {
        let mut cast: Vec<T> = Vec::with_capacity(tuple.len());
        for val in tuple.inner.into_iter() {
            if let Some(val) = val.opt_cast_into() {
                cast.push(val)
            } else {
                return None;
            }
        }

        Some(cast)
    }
}

impl<F, T: TryCastFrom<F>> TryCastFrom<Tuple<F>> for (T,) {
    fn can_cast_from(source: &Tuple<F>) -> bool {
        source.len() == 1 && T::can_cast_from(&source[0])
    }

    fn opt_cast_from(mut source: Tuple<F>) -> Option<(T,)> {
        if source.len() == 1 {
            source.pop().unwrap().opt_cast_into().map(|item| (item,))
        } else {
            None
        }
    }
}

impl<F, T1: TryCastFrom<F>, T2: TryCastFrom<F>> TryCastFrom<Tuple<F>> for (T1, T2) {
    fn can_cast_from(source: &Tuple<F>) -> bool {
        source.len() == 2 && T1::can_cast_from(&source[0]) && T2::can_cast_from(&source[1])
    }

    fn opt_cast_from(mut source: Tuple<F>) -> Option<(T1, T2)> {
        if source.len() == 2 {
            let second: Option<T2> = source.pop().unwrap().opt_cast_into();
            let first: Option<T1> = source.pop().unwrap().opt_cast_into();
            match (first, second) {
                (Some(first), Some(second)) => Some((first, second)),
                _ => None,
            }
        } else {
            None
        }
    }
}

impl<F, T1: TryCastFrom<F>, T2: TryCastFrom<F>, T3: TryCastFrom<F>> TryCastFrom<Tuple<F>>
    for (T1, T2, T3)
{
    fn can_cast_from(source: &Tuple<F>) -> bool {
        source.len() == 3
            && T1::can_cast_from(&source[0])
            && T2::can_cast_from(&source[1])
            && T3::can_cast_from(&source[2])
    }

    fn opt_cast_from(mut source: Tuple<F>) -> Option<(T1, T2, T3)> {
        if source.len() == 3 {
            let third: Option<T3> = source.pop().unwrap().opt_cast_into();
            let second: Option<T2> = source.pop().unwrap().opt_cast_into();
            let first: Option<T1> = source.pop().unwrap().opt_cast_into();
            match (first, second, third) {
                (Some(first), Some(second), Some(third)) => Some((first, second, third)),
                _ => None,
            }
        } else {
            None
        }
    }
}

impl<F, T1: TryCastFrom<F>, T2: TryCastFrom<F>, T3: TryCastFrom<F>, T4: TryCastFrom<F>>
    TryCastFrom<Tuple<F>> for (T1, T2, T3, T4)
{
    fn can_cast_from(source: &Tuple<F>) -> bool {
        source.len() == 4
            && T1::can_cast_from(&source[0])
            && T2::can_cast_from(&source[1])
            && T3::can_cast_from(&source[2])
            && T4::can_cast_from(&source[3])
    }

    fn opt_cast_from(mut source: Tuple<F>) -> Option<(T1, T2, T3, T4)> {
        if source.len() == 4 {
            let fourth: Option<T4> = source.pop().unwrap().opt_cast_into();
            let third: Option<T3> = source.pop().unwrap().opt_cast_into();
            let second: Option<T2> = source.pop().unwrap().opt_cast_into();
            let first: Option<T1> = source.pop().unwrap().opt_cast_into();
            match (first, second, third, fourth) {
                (Some(first), Some(second), Some(third), Some(fourth)) => {
                    Some((first, second, third, fourth))
                }
                _ => None,
            }
        } else {
            None
        }
    }
}

#[async_trait]
impl<T: FromStream> FromStream for Tuple<T>
where
    T::Context: Copy,
{
    type Context = T::Context;

    async fn from_stream<D: Decoder>(context: Self::Context, d: &mut D) -> Result<Self, D::Error> {
        let inner = Vec::<T>::from_stream(context, d).await?;
        Ok(Self { inner })
    }
}

impl<'en, T: 'en> IntoStream<'en> for Tuple<T>
where
    T: IntoStream<'en>,
{
    fn into_stream<E: Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        self.inner.into_stream(encoder)
    }
}

impl<'en, T: 'en> ToStream<'en> for Tuple<T>
where
    T: ToStream<'en>,
{
    fn to_stream<E: Encoder<'en>>(&'en self, encoder: E) -> Result<E::Ok, E::Error> {
        self.inner.to_stream(encoder)
    }
}

impl<T: fmt::Debug> fmt::Debug for Tuple<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "({})",
            self.inner
                .iter()
                .map(|item| format!("{:?}", item))
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}

impl<T: fmt::Display> fmt::Display for Tuple<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "({})",
            self.inner
                .iter()
                .map(|item| item.to_string())
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}
