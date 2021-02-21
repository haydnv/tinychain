//! A generic [`Id`]

use std::fmt;
use std::iter;
use std::ops::Deref;
use std::path::PathBuf;
use std::str::FromStr;

use async_trait::async_trait;
use destream::{de, Decoder, Encoder, FromStream, IntoStream, ToStream};
use regex::Regex;
use safecast::TryCastFrom;
use serde::de::{Deserialize, Deserializer, Error};
use serde::ser::{Serialize, Serializer};

use tc_error::*;

const RESERVED_CHARS: [&str; 21] = [
    "/", "..", "~", "$", "`", "&", "|", "=", "^", "{", "}", "<", ">", "'", "\"", "?", ":", "@",
    "#", "(", ")",
];

/// A static label which implements `Into<Id>`.
pub struct Label {
    id: &'static str,
}

/// Return a [`Label`] with the given static `str`.
pub const fn label(id: &'static str) -> Label {
    Label { id }
}

impl fmt::Display for Label {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(self.id)
    }
}

impl From<Label> for Id {
    fn from(l: Label) -> Id {
        Id {
            id: l.id.to_string(),
        }
    }
}

impl From<uuid::Uuid> for Id {
    fn from(id: uuid::Uuid) -> Self {
        Id { id: id.to_string() }
    }
}

/// A generic Id
///
/// Id is widely used within the Tinychain host software to identify individual variables
/// within a transaction context as well as files and directories.
///
/// An Id must be valid UTF8 and must not contain whitespace or any control character sequence like
/// `{/, .., ~, $, \`, ^, &, |, =, {, }, <, >, ', ", ?, :, @, #}`.
#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Id {
    id: String,
}

impl Id {
    /// Borrows the String underlying this Id.
    pub fn as_str(&self) -> &str {
        self.id.as_str()
    }

    /// Return true if this ID begins with the specified string.
    pub fn starts_with(&self, prefix: &str) -> bool {
        self.id.starts_with(prefix)
    }
}

impl PartialEq<str> for Id {
    fn eq(&self, other: &str) -> bool {
        self.id == other
    }
}

impl<'a> PartialEq<&'a str> for Id {
    fn eq(&self, other: &&'a str) -> bool {
        self.id == *other
    }
}

impl PartialEq<Label> for Id {
    fn eq(&self, other: &Label) -> bool {
        self.id == other.id
    }
}

impl PartialEq<Id> for &str {
    fn eq(&self, other: &Id) -> bool {
        self == &other.id
    }
}

impl From<usize> for Id {
    fn from(u: usize) -> Id {
        u.to_string().parse().unwrap()
    }
}

impl From<u64> for Id {
    fn from(i: u64) -> Id {
        i.to_string().parse().unwrap()
    }
}

#[async_trait]
impl FromStream for Id {
    type Context = ();

    async fn from_stream<D: Decoder>(_context: (), d: &mut D) -> Result<Self, D::Error> {
        d.decode_any(IdVisitor).await
    }
}

struct IdVisitor;

#[async_trait]
impl de::Visitor for IdVisitor {
    type Value = Id;

    fn expecting() -> &'static str {
        "a Tinychain Id like {\"foo\": []}"
    }

    fn visit_string<E: de::Error>(self, s: String) -> Result<Self::Value, E> {
        Id::from_str(&s).map_err(de::Error::custom)
    }

    async fn visit_map<M: de::MapAccess>(self, mut access: M) -> Result<Self::Value, M::Error> {
        if let Some(key) = access.next_key::<String>(()).await? {
            let value: [u8; 0] = access.next_value(()).await?;
            if value.is_empty() {
                Id::from_str(&key).map_err(de::Error::custom)
            } else {
                Err(de::Error::custom("Expected Id but found OpRef"))
            }
        } else {
            Err(de::Error::custom("Unable to parse Id"))
        }
    }
}

impl<'en> ToStream<'en> for Id {
    fn to_stream<E: Encoder<'en>>(&'en self, e: E) -> Result<E::Ok, E::Error> {
        e.encode_str(&self.id)
    }
}

impl<'en> IntoStream<'en> for Id {
    fn into_stream<E: Encoder<'en>>(self, e: E) -> Result<E::Ok, E::Error> {
        e.encode_str(&self.id)
    }
}

impl FromStr for Id {
    type Err = TCError;

    fn from_str(id: &str) -> TCResult<Id> {
        validate_id(id)?;
        Ok(Id { id: id.to_string() })
    }
}

impl TryCastFrom<String> for Id {
    fn can_cast_from(id: &String) -> bool {
        validate_id(id).is_ok()
    }

    fn opt_cast_from(id: String) -> Option<Id> {
        id.parse().ok()
    }
}

impl TryCastFrom<Id> for usize {
    fn can_cast_from(id: &Id) -> bool {
        id.as_str().parse::<usize>().is_ok()
    }

    fn opt_cast_from(id: Id) -> Option<usize> {
        id.as_str().parse::<usize>().ok()
    }
}

impl fmt::Display for Id {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.id)
    }
}

fn validate_id(id: &str) -> TCResult<()> {
    if id.is_empty() {
        return Err(TCError::bad_request("Id cannot be empty", id));
    }

    let filtered: &str = &id.chars().filter(|c| *c as u8 > 32).collect::<String>();
    if filtered != id {
        return Err(TCError::bad_request(
            "This value ID contains an ASCII control character",
            filtered,
        ));
    }

    for pattern in &RESERVED_CHARS {
        if id.contains(pattern) {
            return Err(TCError::bad_request(
                "A value ID may not contain this pattern",
                pattern,
            ));
        }
    }

    if let Some(w) = Regex::new(r"\s").unwrap().find(id) {
        return Err(TCError::bad_request(
            "A value ID may not contain whitespace",
            format!("{:?}", w),
        ));
    }

    Ok(())
}

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

/// A Tinychain path.
#[derive(Clone, Debug, Default, Hash, Eq, PartialEq)]
pub struct TCPathBuf {
    segments: Vec<PathSegment>,
}

impl TCPathBuf {
    /// Return a mutable reference to the underlying vector.
    pub fn as_mut(&'_ mut self) -> &'_ mut Vec<PathSegment> {
        &mut self.segments
    }

    /// Return a complete slice of the underlying vector.
    pub fn as_slice(&'_ self) -> &'_ [PathSegment] {
        &self.segments[..]
    }

    /// Consumes `self` and returns its underlying vector.
    pub fn into_vec(self) -> Vec<PathSegment> {
        self.segments
    }

    /// Appends `suffix` to this `TCPathBuf`.
    pub fn append<T: Into<PathSegment>>(mut self, suffix: T) -> Self {
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

impl Extend<PathSegment> for TCPathBuf {
    fn extend<T: IntoIterator<Item = PathSegment>>(&mut self, iter: T) {
        self.segments.extend(iter)
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
    type Target = [PathSegment];

    fn deref(&'_ self) -> &'_ [PathSegment] {
        &self.segments[..]
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
                .collect::<TCResult<Vec<PathSegment>>>()?;

            Ok(TCPathBuf { segments })
        } else {
            Ok(TCPathBuf {
                segments: iter::once(to.parse()?).collect(),
            })
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
        Self::from_str(&path).map_err(D::Error::custom)
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

/// A borrowed Tinychain path which implements [`fmt::Display`].
pub struct TCPath<'a> {
    inner: &'a [PathSegment],
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
