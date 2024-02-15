use serde::de::{Deserialize, Deserializer};
use serde::ser::{Serialize, Serializer};
use umask::Mode;

use tcgeneric::TCPathBuf;

#[derive(Clone)]
pub struct Claim {
    path: TCPathBuf,
    mode: Mode,
}

impl Claim {
    pub fn new<Path: Into<TCPathBuf>>(path: Path, mode: Mode) -> Self {
        Self {
            path: path.into(),
            mode,
        }
    }
}

impl<'de> Deserialize<'de> for Claim {
    fn deserialize<D>(deserializer: D) -> Result<Claim, D::Error>
    where
        D: Deserializer<'de>,
    {
        <(TCPathBuf, u32) as Deserialize<'de>>::deserialize(deserializer).map(|(path, mode)| {
            Claim {
                path,
                mode: mode.into(),
            }
        })
    }
}

impl Serialize for Claim {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        (&self.path, u32::from(self.mode)).serialize(serializer)
    }
}
