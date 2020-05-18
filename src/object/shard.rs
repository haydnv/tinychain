use std::convert::{TryFrom, TryInto};
use std::fmt;

use async_trait::async_trait;

use crate::error;
use crate::object::TCObject;
use crate::value::{TCPath, TCResult, TCValue};

pub struct Shard {
    hosted: TCPath,
}

impl fmt::Display for Shard {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Shard()")
    }
}

impl From<Shard> for TCValue {
    fn from(shard: Shard) -> TCValue {
        TCValue::Vector(vec![shard.hosted.into()])
    }
}

impl TryFrom<TCValue> for Shard {
    type Error = error::TCError;

    fn try_from(value: TCValue) -> TCResult<Shard> {
        let mut value: Vec<TCValue> = value.try_into()?;
        if value.len() == 1 {
            let hosted = value.pop().unwrap().try_into()?;
            Ok(Shard { hosted })
        } else {
            let value: TCValue = value.into();
            Err(error::bad_request("Expected Shard, found", value))
        }
    }
}

#[async_trait]
impl TCObject for Shard {
    fn class() -> &'static str {
        "Shard"
    }

    fn id(&self) -> TCValue {
        self.hosted.clone().into()
    }
}
