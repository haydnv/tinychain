use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::path::PathBuf;

use serde::de;
use serde::ser::Serializer;

use crate::error;
use crate::value::{TCResult, ValueId};

pub type PathSegment = ValueId;

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct TCPath {
    segments: Vec<PathSegment>,
}

impl TCPath {
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    pub fn len(&self) -> usize {
        self.segments.len()
    }

    pub fn slice_from(&self, start: usize) -> TCPath {
        TCPath {
            segments: self.segments[start..].to_vec(),
        }
    }

    pub fn slice_to(&self, end: usize) -> TCPath {
        TCPath {
            segments: self.segments[..end].to_vec(),
        }
    }

    pub fn starts_with(&self, other: TCPath) -> bool {
        if self.len() < other.len() {
            false
        } else {
            self.segments[0..other.len()] == other.segments[..]
        }
    }

    pub fn push(&mut self, segment: PathSegment) {
        self.segments.push(segment)
    }
}

impl Extend<PathSegment> for TCPath {
    fn extend<T: IntoIterator<Item = PathSegment>>(&mut self, iter: T) {
        self.segments.extend(iter)
    }
}

impl IntoIterator for TCPath {
    type Item = PathSegment;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.segments.into_iter()
    }
}

impl<Idx> std::ops::Index<Idx> for TCPath
where
    Idx: std::slice::SliceIndex<[PathSegment]>,
{
    type Output = Idx::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.segments[index]
    }
}

impl fmt::Display for TCPath {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "/{}",
            self.segments
                .iter()
                .map(String::from)
                .collect::<Vec<String>>()
                .join("/")
        )
    }
}

impl PartialEq<ValueId> for TCPath {
    fn eq(&self, other: &ValueId) -> bool {
        if self.len() == 1 {
            &self.segments[0] == other
        } else {
            false
        }
    }
}

impl PartialEq<TCPath> for ValueId {
    fn eq(&self, other: &TCPath) -> bool {
        if other.len() == 1 {
            &other.segments[0] == self
        } else {
            false
        }
    }
}

impl From<Vec<PathSegment>> for TCPath {
    fn from(segments: Vec<PathSegment>) -> TCPath {
        TCPath { segments }
    }
}

impl From<PathSegment> for TCPath {
    fn from(segment: PathSegment) -> TCPath {
        TCPath {
            segments: vec![segment],
        }
    }
}

impl From<TCPath> for PathBuf {
    fn from(path: TCPath) -> PathBuf {
        PathBuf::from(format!("{}", path))
    }
}

impl TryFrom<&str> for TCPath {
    type Error = error::TCError;

    fn try_from(to: &str) -> TCResult<TCPath> {
        if to == "/" {
            Ok(TCPath { segments: vec![] })
        } else if to.ends_with('/') {
            Err(error::bad_request("Path cannot end with a slash", to))
        } else if to.starts_with('/') {
            let segments: Vec<&str> = to.split('/').collect();
            let segments = &segments[1..];
            Ok(TCPath {
                segments: segments
                    .iter()
                    .cloned()
                    .map(PathSegment::try_from)
                    .collect::<TCResult<Vec<PathSegment>>>()?,
            })
        } else {
            Ok(TCPath {
                segments: vec![to.try_into()?],
            })
        }
    }
}

impl TryFrom<String> for TCPath {
    type Error = error::TCError;

    fn try_from(s: String) -> TCResult<TCPath> {
        TCPath::try_from(s.as_str())
    }
}

impl<'de> serde::Deserialize<'de> for TCPath {
    fn deserialize<D>(deserializer: D) -> Result<TCPath, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        let s: &str = de::Deserialize::deserialize(deserializer)?;
        s.try_into().map_err(de::Error::custom)
    }
}

impl serde::Serialize for TCPath {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        s.serialize_str(&format!("{}", self))
    }
}
