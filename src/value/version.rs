use std::fmt;
use std::str::FromStr;

use serde::de::{Deserialize, Deserializer, Error};
use serde::ser::{Serialize, Serializer};

use crate::error;
use crate::value::TCResult;

#[derive(Clone)]
pub struct Version {
    major: u64,
    minor: u64,
    patch: u64,
}

impl FromStr for Version {
    type Err = error::TCError;

    fn from_str(v: &str) -> TCResult<Version> {
        let fields: Vec<u64> = v
            .split('.')
            .map(|s| {
                s.parse::<u64>()
                    .map_err(|e| error::bad_request("Unable to parse version number", e))
            })
            .collect::<TCResult<Vec<u64>>>()?;
        if fields.len() != 3 {
            Err(error::bad_request(
                "Expected a version of the form \"<major>.<minor>.<patch>\", found {}",
                v,
            ))
        } else {
            Ok(Version {
                major: fields[0],
                minor: fields[1],
                patch: fields[2],
            })
        }
    }
}

impl<'de> Deserialize<'de> for Version {
    fn deserialize<D>(deserializer: D) -> Result<Version, D::Error>
    where
        D: Deserializer<'de>,
    {
        Version::from_str(Deserialize::deserialize(deserializer)?).map_err(Error::custom)
    }
}

impl Serialize for Version {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        s.serialize_str(&format!("{}", self))
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}
