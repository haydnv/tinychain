use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use ed25519_dalek::{Keypair, PublicKey, SecretKey, Signature};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};

use crate::error;
use crate::object::TCObject;
use crate::state::table;
use crate::value::link::{Link, TCPath};
use crate::value::{Op, TCResult, TCValue};

#[derive(Clone, Deserialize, Serialize)]
pub struct Token {
    iss: Link,
    iat: u64,
    exp: u64,
    actor_id: TCValue,
    scopes: Vec<TCPath>,
}

impl Token {
    pub fn actor_id(&self) -> Op {
        Op::Get {
            subject: self.iss.clone().into(),
            key: Box::new(self.actor_id.clone()),
        }
    }

    pub fn get_actor(token: &str) -> TCResult<Op> {
        let token: Vec<&str> = token.split('.').collect();
        if token.len() != 3 {
            return Err(error::unauthorized(
                "Expected bearer token in the format '<header>.<claims>.<data>'",
            ));
        }

        let token = base64::decode(token[1])
            .map_err(|e| error::unauthorized(&format!("Invalid bearer token: {}", e)))?;
        let token: Token = serde_json::from_slice(&token)
            .map_err(|e| error::unauthorized(&format!("Invalid bearer token: {}", e)))?;

        Ok(Op::Get {
            subject: token.iss.into(),
            key: Box::new(token.actor_id),
        })
    }
}

pub struct Actor {
    host: Link,
    id: TCValue,
    lock: Op,
    public_key: PublicKey,
    private_key: SecretKey,
}

impl Actor {
    pub fn new(host: Link, id: TCValue, lock: Op) -> Arc<Actor> {
        let mut rng = OsRng {};
        let keypair: Keypair = Keypair::generate(&mut rng);

        Arc::new(Actor {
            host,
            id,
            lock,
            public_key: keypair.public,
            private_key: keypair.secret,
        })
    }

    pub fn sign_token(&self, token: &Token) -> TCResult<String> {
        let keypair = Keypair::from_bytes(
            &[self.private_key.to_bytes(), self.public_key.to_bytes()].concat(),
        )
        .map_err(|_| error::unauthorized("Unable to construct ECDSA keypair for the given user"))?;

        let header = base64::encode(serde_json::to_string_pretty(&TokenHeader::default()).unwrap());
        let claims = base64::encode(serde_json::to_string_pretty(&token).unwrap());
        let signature = base64::encode(
            &keypair
                .sign(format!("{}.{}", header, claims).as_bytes())
                .to_bytes()[..],
        );
        Ok(format!("{}.{}.{}", header, claims, signature))
    }

    pub fn validate(&self, encoded: &str) -> TCResult<Token> {
        let mut encoded: Vec<&str> = encoded.split('.').collect();
        if encoded.len() != 3 {
            return Err(error::unauthorized(
                "Expected bearer token in the format '<header>.<claims>.<data>'",
            ));
        }

        let message = format!("{}.{}", encoded[0], encoded[1]);
        let signature =
            Signature::from_bytes(&base64::decode(encoded.pop().unwrap()).map_err(|e| {
                error::unauthorized(&format!("Invalid bearer token signature: {}", e))
            })?)
            .map_err(|_| error::unauthorized("Invalid bearer token signature"))?;

        let token = encoded.pop().unwrap();
        let token = base64::decode(token)
            .map_err(|e| error::unauthorized(&format!("Invalid bearer token: {}", e)))?;
        let token: Token = serde_json::from_slice(&token)
            .map_err(|e| error::unauthorized(&format!("Invalid bearer token: {}", e)))?;

        if token.actor_id != self.id {
            return Err(error::unauthorized(
                "Attempted to use a bearer token for a different user",
            ));
        }

        let header = encoded.pop().unwrap();
        let header = base64::decode(header)
            .map_err(|e| error::unauthorized(&format!("Invalid bearer token header: {}", e)))?;
        let header: TokenHeader = serde_json::from_slice(&header)
            .map_err(|e| error::unauthorized(&format!("Invalid bearer token header: {}", e)))?;

        if header != TokenHeader::default() {
            Err(error::unauthorized("Unsupported bearer token type"))
        } else if self
            .public_key
            .verify(message.as_bytes(), &signature)
            .is_err()
        {
            Err(error::unauthorized("Invalid bearer token provided"))
        } else {
            Ok(token)
        }
    }
}

impl Into<table::Row> for Actor {
    fn into(self) -> table::Row {
        table::Row::from(
            vec![self.host.into(), self.id],
            vec![
                Some(self.public_key.to_bytes().to_vec().into()),
                Some(self.private_key.to_bytes().to_vec().into()),
            ],
        )
    }
}

impl TryFrom<table::Row> for Actor {
    type Error = error::TCError;

    fn try_from(row: table::Row) -> TCResult<Actor> {
        let (mut key, mut values) = row.into();

        if key.len() != 2 || values.len() != 2 {
            Err(error::bad_request(
                "Expected Actor, found",
                format!("{:?}: {:?}", key, values),
            ))
        } else {
            let id = key.pop().unwrap().try_into()?;
            let host = key.pop().unwrap().try_into()?;
            let private_key: Bytes = values
                .pop()
                .unwrap()
                .ok_or_else(|| error::bad_request("Actor::from(Row) missing field", "private_key"))?
                .try_into()?;
            let public_key: Bytes = values
                .pop()
                .unwrap()
                .ok_or_else(|| error::bad_request("Actor::from(Row) missing field", "public_key"))?
                .try_into()?;
            let lock = values
                .pop()
                .unwrap()
                .ok_or_else(|| error::bad_request("Actor::from(Row) missing field", "lock"))?
                .try_into()?;

            if values.is_empty() {
                Ok(Actor {
                    host,
                    id,
                    lock,
                    public_key: PublicKey::from_bytes(&public_key[..])
                        .map_err(|e| error::bad_request("Unable to parse public key", e))?,
                    private_key: SecretKey::from_bytes(&private_key[..])
                        .map_err(|e| error::bad_request("Unable to parse private key", e))?,
                })
            } else {
                Err(error::bad_request(
                    "Got extra unknown values for Actor",
                    format!("{:?}", values),
                ))
            }
        }
    }
}

impl fmt::Display for Actor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Actor({}, {})", self.host, self.id)
    }
}

#[async_trait]
impl TCObject for Actor {
    fn class() -> &'static str {
        "Actor"
    }
}

#[derive(Deserialize, Serialize, Eq, PartialEq)]
struct TokenHeader {
    alg: String,
    typ: String,
}

impl Default for TokenHeader {
    fn default() -> TokenHeader {
        TokenHeader {
            alg: "ES256".into(),
            typ: "JWT".into(),
        }
    }
}
