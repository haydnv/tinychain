use std::convert::{TryFrom, TryInto};

use bytes::Bytes;
use ed25519_dalek::Keypair;
use rand::rngs::OsRng;

use crate::error;
use crate::host::NetworkTime;
use crate::value::{Link, TCResult, TCValue};

pub struct Actor {
    private_key: Bytes,
    public_key: Bytes,
}

impl Actor {
    pub fn new() -> Actor {
        let mut rng = OsRng {};
        let keypair: Keypair = Keypair::generate(&mut rng);

        Actor {
            private_key: Bytes::copy_from_slice(&keypair.secret.to_bytes()[..]),
            public_key: Bytes::copy_from_slice(&keypair.public.to_bytes()[..]),
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

struct TokenHeader {
    alg: String,
    typ: String,
}

struct TokenClaims {
    iss: Link,
    iat: u64,
    exp: u64,
}

pub struct Token {
    header: TokenHeader,
    claims: TokenClaims,
}

impl Token {
    pub fn new(issuer: Link, issued_at: NetworkTime, expires: NetworkTime) -> Token {
        Token {
            header: TokenHeader {
                alg: "ES256".into(),
                typ: "JWT".into(),
            },
            claims: TokenClaims {
                iss: issuer,
                iat: issued_at.as_millis(),
                exp: expires.as_millis(),
            },
        }
    }
}
