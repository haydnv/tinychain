//! A generic [`Id`]

use std::fmt;
use std::iter;
use std::ops::{Deref, DerefMut};
use std::path::PathBuf;
use std::str::FromStr;

use async_hash::Hash;
use async_trait::async_trait;
use destream::de::{self, Decoder, FromStream};
use destream::en::{Encoder, IntoStream, ToStream};
use safecast::TryCastFrom;
use serde::de::{Deserialize, Deserializer};
use serde::ser::{Serialize, Serializer};
use sha2::digest::{Digest, Output};

use tc_error::*;

pub use hr_id::{label, Id, Label};

/// An alias for [`Id`] used for code clarity.
pub type PathSegment = Id;

/// A constant representing a [`TCPathBuf`].
pub struct PathLabel {
    segments: &'static [&'static str],
}

impl<Idx: std::slice::SliceIndex<[&'static str]>> std::ops::Index<Idx> for PathLabel {
    type Output = Idx::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.segments[index]
    }
}

/// Return a [`PathLabel`] with the given segments.
pub const fn path_label(segments: &'static [&'static str]) -> PathLabel {
    PathLabel { segments }
}

impl From<PathLabel> for TCPathBuf {
    fn from(path: PathLabel) -> Self {
        let segments = path
            .segments
            .into_iter()
            .map(|segment| label(*segment))
            .map(PathSegment::from)
            .collect();

        Self { segments }
    }
}

/// A TinyChain path.
#[derive(Clone, Debug, Default, Hash, Eq, PartialEq)]
pub struct TCPathBuf {
    segments: Vec<PathSegment>,
}

impl TCPathBuf {
    /// Return a complete slice of the underlying vector.
    pub fn as_slice(&'_ self) -> &'_ [PathSegment] {
        &self.segments[..]
    }

    /// Consumes `self` and returns its underlying vector.
    pub fn into_vec(self) -> Vec<PathSegment> {
        self.segments
    }

    /// Appends `suffix` to this `TCPathBuf`.
    pub fn append<S: Into<PathSegment>>(mut self, suffix: S) -> Self {
        self.segments.push(suffix.into());
        self
    }

    /// If this path begins with the specified prefix, returns the suffix following the prefix.
    pub fn suffix<'a>(&self, path: &'a [PathSegment]) -> Option<&'a [PathSegment]> {
        if path.starts_with(&self.segments) {
            Some(&path[self.segments.len()..])
        } else {
            None
        }
    }
}

impl<Idx: std::slice::SliceIndex<[PathSegment]>> std::ops::Index<Idx> for TCPathBuf {
    type Output = Idx::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.segments[index]
    }
}

impl IntoIterator for TCPathBuf {
    type Item = PathSegment;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.segments.into_iter()
    }
}

impl std::borrow::Borrow<[PathSegment]> for TCPathBuf {
    fn borrow(&self) -> &[PathSegment] {
        &self.segments[..]
    }
}

impl Deref for TCPathBuf {
    type Target = Vec<PathSegment>;

    fn deref(&self) -> &Vec<PathSegment> {
        &self.segments
    }
}

impl DerefMut for TCPathBuf {
    fn deref_mut(&mut self) -> &mut Vec<PathSegment> {
        &mut self.segments
    }
}

impl PartialEq<[PathSegment]> for TCPathBuf {
    fn eq(&self, other: &[PathSegment]) -> bool {
        &self.segments == other
    }
}

impl<D: Digest> Hash<D> for TCPathBuf {
    fn hash(self) -> Output<D> {
        Hash::<D>::hash(self.to_string())
    }
}

impl<'a, D: Digest> Hash<D> for &'a TCPathBuf {
    fn hash(self) -> Output<D> {
        Hash::<D>::hash(self.to_string())
    }
}

impl From<PathSegment> for TCPathBuf {
    fn from(segment: PathSegment) -> TCPathBuf {
        TCPathBuf {
            segments: iter::once(segment).collect(),
        }
    }
}

impl From<Label> for TCPathBuf {
    fn from(segment: Label) -> TCPathBuf {
        TCPathBuf {
            segments: iter::once(segment.into()).collect(),
        }
    }
}

impl From<TCPathBuf> for PathBuf {
    fn from(path: TCPathBuf) -> PathBuf {
        PathBuf::from(path.to_string())
    }
}

impl FromStr for TCPathBuf {
    type Err = TCError;

    #[inline]
    fn from_str(to: &str) -> TCResult<TCPathBuf> {
        if to == "/" {
            Ok(TCPathBuf { segments: vec![] })
        } else if to.ends_with('/') {
            Err(TCError::bad_request("Path cannot end with a slash", to))
        } else if to.starts_with('/') {
            let segments = to
                .split('/')
                .skip(1)
                .map(PathSegment::from_str)
                .map(|r| r.map_err(TCError::unsupported))
                .collect::<TCResult<Vec<PathSegment>>>()?;

            Ok(TCPathBuf { segments })
        } else {
            to.parse()
                .map(|id| TCPathBuf {
                    segments: iter::once(id).collect(),
                })
                .map_err(TCError::unsupported)
        }
    }
}

impl From<Vec<PathSegment>> for TCPathBuf {
    fn from(segments: Vec<PathSegment>) -> Self {
        Self { segments }
    }
}

impl iter::FromIterator<PathSegment> for TCPathBuf {
    fn from_iter<T: IntoIterator<Item = PathSegment>>(iter: T) -> Self {
        TCPathBuf {
            segments: iter.into_iter().collect(),
        }
    }
}

impl<'de> Deserialize<'de> for TCPathBuf {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let path = String::deserialize(deserializer)?;
        Self::from_str(&path).map_err(serde::de::Error::custom)
    }
}

impl Serialize for TCPathBuf {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.to_string().serialize(serializer)
    }
}

#[async_trait]
impl FromStream for TCPathBuf {
    type Context = ();

    async fn from_stream<D: Decoder>(context: (), decoder: &mut D) -> Result<TCPathBuf, D::Error> {
        let s = String::from_stream(context, decoder).await?;
        s.parse().map_err(de::Error::custom)
    }
}

impl<'en> IntoStream<'en> for TCPathBuf {
    fn into_stream<E: Encoder<'en>>(self, e: E) -> Result<E::Ok, E::Error> {
        e.encode_str(&self.to_string())
    }
}

impl<'en> ToStream<'en> for TCPathBuf {
    fn to_stream<E: Encoder<'en>>(&'en self, e: E) -> Result<E::Ok, E::Error> {
        e.encode_str(&self.to_string())
    }
}

impl TryCastFrom<TCPathBuf> for Id {
    fn can_cast_from(path: &TCPathBuf) -> bool {
        path.as_slice().len() == 1
    }

    fn opt_cast_from(path: TCPathBuf) -> Option<Id> {
        let mut segments = path.into_vec();
        if segments.len() == 1 {
            segments.pop()
        } else {
            None
        }
    }
}

impl fmt::Display for TCPathBuf {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", TCPath::from(&self[..]))
    }
}

/// A borrowed TinyChain path which implements [`fmt::Display`].
pub struct TCPath<'a> {
    inner: &'a [PathSegment],
}

impl Default for TCPath<'static> {
    fn default() -> Self {
        Self { inner: &[] }
    }
}

impl<'a> From<&'a [PathSegment]> for TCPath<'a> {
    fn from(inner: &'a [PathSegment]) -> TCPath<'a> {
        TCPath { inner }
    }
}

impl<'a, Idx: std::slice::SliceIndex<[PathSegment]>> std::ops::Index<Idx> for TCPath<'a> {
    type Output = Idx::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.inner[index]
    }
}

impl<'a> fmt::Display for TCPath<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "/{}",
            self.inner
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<String>>()
                .join("/")
        )
    }
}
