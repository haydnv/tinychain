use std::cmp::Ordering;
use std::fmt;
use std::str::FromStr;

use async_hash::{Digest, Hash, Output};
use async_trait::async_trait;
use destream::{de, en};
use get_size::GetSize;
use get_size_derive::*;
use safecast::TryCastFrom;
use serde::de::{Deserialize, Deserializer};
use serde::ser::{Serialize, Serializer};

use tc_error::*;
use tcgeneric::Id;

/// A semantic version with a major, minor, and patch number, e.g. "0.1.12"
#[derive(Clone, Copy, Default, std::hash::Hash, Eq, PartialEq, GetSize)]
pub struct Version {
    major: u32,
    minor: u32,
    patch: u32,
}

impl Version {
    /// Construct an [`Id`] from this [`Version`] number.
    pub fn to_id(&self) -> Id {
        Id::try_cast_from(self.to_string(), |_| {
            unreachable!("number failed ID validation")
        })
        .unwrap()
    }
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
                Ordering::Equal => self.patch.cmp(&other.patch),
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
        Hash::<D>::hash([self.major, self.minor, self.patch])
    }
}

impl From<(u32, u32, u32)> for Version {
    fn from(version: (u32, u32, u32)) -> Self {
        let (major, minor, patch) = version;
        Self {
            major,
            minor,
            patch,
        }
    }
}

impl FromStr for Version {
    type Err = TCError;

    fn from_str(s: &str) -> TCResult<Self> {
        if s.find('.').is_none() {
            return Err(TCError::unexpected(s, "a version number"));
        }

        let mut parts = s.split('.');

        let major = if let Some(major) = parts.next() {
            major.parse().map_err(|cause| {
                TCError::unexpected(major, "a major version number").consume(cause)
            })?
        } else {
            return Err(bad_request!(
                "{} is missing missing a major version number",
                s
            ));
        };

        let minor = if let Some(minor) = parts.next() {
            minor.parse().map_err(|cause| {
                TCError::unexpected(minor, "a minor version number").consume(cause)
            })?
        } else {
            return Err(bad_request!(
                "{} is missing missing a minor version number",
                s
            ));
        };

        let patch = if let Some(patch) = parts.next() {
            patch
                .parse()
                .map_err(|cause| TCError::unexpected(patch, "a patch number").consume(cause))?
        } else {
            return Err(bad_request!("{} is missing missing a patch number", s));
        };

        if parts.next().is_some() {
            return Err(bad_request!("invalid semantic version number: {}", s));
        }

        Ok(Self {
            major,
            minor,
            patch: patch,
        })
    }
}

impl TryCastFrom<&str> for Version {
    fn can_cast_from(value: &&str) -> bool {
        Self::from_str(value).is_ok()
    }

    fn opt_cast_from(value: &str) -> Option<Self> {
        Self::from_str(value).ok()
    }
}

impl TryCastFrom<Id> for Version {
    fn can_cast_from(value: &Id) -> bool {
        Self::from_str(value.as_str()).is_ok()
    }

    fn opt_cast_from(value: Id) -> Option<Self> {
        Self::from_str(value.as_str()).ok()
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
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}
