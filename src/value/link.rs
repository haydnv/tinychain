use std::fmt;

use serde::de;
use serde::ser::Serializer;

use crate::context::TCResult;
use crate::error;
use crate::value::{validate_id, TCValueExt};

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct Link {
    segments: Vec<String>,
}

impl Link {
    fn _validate(to: &str) -> TCResult<()> {
        validate_id(to)?;

        if !to.starts_with('/') {
            Err(error::bad_request(
                "Expected an absolute path starting with '/' but found",
                to,
            ))
        } else if to != "/" && to.ends_with('/') {
            Err(error::bad_request("Trailing slash is not allowed", to))
        } else {
            Ok(())
        }
    }

    pub fn to(destination: &str) -> TCResult<Link> {
        Link::_validate(destination)?;

        let segments: Vec<String> = if destination == "/" {
            vec![]
        } else {
            destination[1..].split('/').map(|s| s.to_string()).collect()
        };

        Ok(Link { segments })
    }

    pub fn append(&self, suffix: &Link) -> Link {
        Link::to(&format!("{}{}", self, suffix)).unwrap()
    }

    pub fn as_str(&self, index: usize) -> &str {
        self.segments[index].as_str()
    }

    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    pub fn len(&self) -> usize {
        self.segments.len()
    }

    pub fn nth(&self, i: usize) -> Link {
        Link {
            segments: vec![self.segments[i].clone()],
        }
    }

    pub fn slice_from(&self, start: usize) -> Link {
        Link {
            segments: self.segments[start..].to_vec(),
        }
    }

    pub fn slice_to(&self, end: usize) -> Link {
        Link {
            segments: self.segments[..end].to_vec(),
        }
    }
}

impl TCValueExt for Link {}

impl From<u64> for Link {
    fn from(i: u64) -> Link {
        Link::to(&format!("/{}", i)).unwrap()
    }
}

impl IntoIterator for Link {
    type Item = Link;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        let mut segments = Vec::with_capacity(self.len());
        for i in 0..self.len() {
            segments.push(self.nth(i));
        }
        segments.into_iter()
    }
}

impl PartialEq<str> for Link {
    fn eq(&self, other: &str) -> bool {
        self.to_string().as_str() == other
    }
}

impl<'de> serde::Deserialize<'de> for Link {
    fn deserialize<D>(deserializer: D) -> Result<Link, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        let s: &str = de::Deserialize::deserialize(deserializer)?;
        Link::to(s).map_err(de::Error::custom)
    }
}

impl serde::Serialize for Link {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        s.serialize_str(&format!("{}", self))
    }
}

impl fmt::Display for Link {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", format!("/{}", self.segments.join("/")))
    }
}
