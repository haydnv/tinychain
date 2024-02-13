use serde::de::{Deserialize, Deserializer};
use serde::ser::{Serialize, Serializer};
use umask::Mode;

use tc_value::Link;

#[derive(Clone)]
pub struct Claim {
    link: Link,
    mode: Mode,
}

impl Claim {
    pub fn new(link: Link, mode: Mode) -> Self {
        Self { link, mode }
    }
}

impl<'de> Deserialize<'de> for Claim {
    fn deserialize<D>(deserializer: D) -> Result<Claim, D::Error>
    where
        D: Deserializer<'de>,
    {
        <(Link, u32) as Deserialize<'de>>::deserialize(deserializer).map(|(link, mode)| Claim {
            link,
            mode: mode.into(),
        })
    }
}

impl Serialize for Claim {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        (&self.link, u32::from(self.mode)).serialize(serializer)
    }
}
