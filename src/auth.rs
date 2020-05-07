use std::convert::{TryFrom, TryInto};

use bytes::Bytes;

use crate::error;
use crate::value::{TCResult, TCValue};

pub struct Actor {
    private_key: Bytes,
    public_key: Bytes,
}

impl Actor {
    pub fn new() -> Actor {
        Actor {
            private_key: Bytes::new(), // TODO
            public_key: Bytes::new(),  // TODO
        }
    }
}

impl From<Actor> for TCValue {
    fn from(actor: Actor) -> TCValue {
        vec![actor.private_key, actor.public_key].into()
    }
}

impl TryFrom<TCValue> for Actor {
    type Error = error::TCError;

    fn try_from(value: TCValue) -> TCResult<Actor> {
        let mut value: Vec<Bytes> = value.try_into()?;
        if value.len() == 2 {
            Ok(Actor {
                private_key: value.remove(0),
                public_key: value.remove(0),
            })
        } else {
            let value: TCValue = value.into();
            Err(error::bad_request("Expected Actor, found", value))
        }
    }
}
