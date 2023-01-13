use std::cmp::Ordering;
use std::fmt;
use std::str::FromStr;

use async_hash::Hash;
use async_trait::async_trait;
use destream::{de, en};
use serde::de::{Deserialize, Deserializer};
use serde::ser::{Serialize, Serializer};
use sha2::digest::{Digest, Output};

use tc_error::*;
use tcgeneric::Id;

/// A semantic version with a major, minor, and revision number, e.g. "0.1.12"
#[derive(Clone, Copy, Default, std::hash::Hash, Eq, PartialEq)]
pub struct Version {
    major: u32,
    minor: u32,
    rev: u32,
}

impl PartialOrd for Version {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Version {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.major.cmp(&other.major) {
            Ordering::Equal => match self.minor.cmp(&other.minor) {
                Ordering::Equal => self.rev.cmp(&other.rev),
                ordering => ordering,
            },
            ordering => ordering,
        }
    }
}

impl PartialEq<String> for Version {
    fn eq(&self, other: &String) -> bool {
        &self.to_string() == other
    }
}

impl PartialEq<str> for Version {
    fn eq(&self, other: &str) -> bool {
        self.to_string().as_str() == other
    }
}

impl<D: Digest> Hash<D> for Version {
    fn hash(self) -> Output<D> {
        Hash::<D>::hash([self.major, self.minor, self.rev])
    }
}

impl From<(u32, u32, u32)> for Version {
    fn from(version: (u32, u32, u32)) -> Self {
        let (major, minor, rev) = version;
        Self { major, minor, rev }
    }
}

impl FromStr for Version {
    type Err = TCError;

    fn from_str(s: &str) -> TCResult<Self> {
        if s.find('.').is_none() {
            return Err(TCError::bad_request("not a valid version number", s));
        }

        let mut parts = s.split('.');

        let major = if let Some(major) = parts.next() {
            major
                .parse()
                .map_err(|err| TCError::bad_request("invalid major version", err))?
        } else {
            return Err(TCError::unsupported("missing major version number"));
        };

        let minor = if let Some(minor) = parts.next() {
            minor
                .parse()
                .map_err(|err| TCError::bad_request("invalid minor version", err))?
        } else {
            return Err(TCError::unsupported("missing minor version number"));
        };

        let rev = if let Some(rev) = parts.next() {
            rev.parse()
                .map_err(|err| TCError::bad_request("invalid revision number", err))?
        } else {
            return Err(TCError::unsupported("missing revision number"));
        };

        if parts.next().is_some() {
            return Err(TCError::unsupported("invalid semantic version"));
        }

        Ok(Self { major, minor, rev })
    }
}

impl From<Version> for Id {
    fn from(version: Version) -> Id {
        version.to_string().parse().expect("version id")
    }
}

impl<'de> Deserialize<'de> for Version {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let version: &str = Deserialize::deserialize(deserializer)?;
        version.parse().map_err(serde::de::Error::custom)
    }
}

impl Serialize for Version {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.to_string().serialize(serializer)
    }
}

#[async_trait]
impl de::FromStream for Version {
    type Context = ();

    async fn from_stream<D: de::Decoder>(cxt: (), decoder: &mut D) -> Result<Self, D::Error> {
        let version = String::from_stream(cxt, decoder).await?;
        version.parse().map_err(de::Error::custom)
    }
}

impl<'en> en::IntoStream<'en> for Version {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        self.to_string().into_stream(encoder)
    }
}

impl<'en> en::ToStream<'en> for Version {
    fn to_stream<E: en::Encoder<'en>>(&self, encoder: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream(self.to_string(), encoder)
    }
}

impl fmt::Debug for Version {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.rev)
    }
}
