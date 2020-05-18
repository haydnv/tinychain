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
use crate::transaction::Txn;
use crate::value::{Link, Op, TCPath, TCResult, TCValue};

#[derive(Deserialize, Serialize)]
pub struct Token {
    iss: Link,
    iat: u64,
    exp: u64,
    actor_id: TCValue,
    scopes: Vec<TCPath>,
}

impl Token {
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
    public_key: PublicKey,
    private_key: Option<SecretKey>,
}

impl Actor {
    pub fn new(_txn: Arc<Txn>, host: Link, id: TCValue) -> Arc<Actor> {
        let mut rng = OsRng {};
        let keypair: Keypair = Keypair::generate(&mut rng);

        Arc::new(Actor {
            host,
            id,
            public_key: keypair.public,
            private_key: Some(keypair.secret),
        })
    }

    pub fn sign_token(&self, token: &Token) -> TCResult<String> {
        let keypair = if let Some(secret) = &self.private_key {
            Keypair::from_bytes(&[secret.to_bytes(), self.public_key.to_bytes()].concat()).map_err(
                |_| error::unauthorized("Unable to construct ECDSA keypair for the given user"),
            )?
        } else {
            return Err(error::forbidden(
                "You are not authorized to issue tokens on behalf of this user".into(),
            ));
        };

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

impl fmt::Display for Actor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Actor({}, {})", self.host, self.id)
    }
}

impl From<Actor> for TCValue {
    fn from(actor: Actor) -> TCValue {
        let private_key: TCValue = if let Some(private_key) = actor.private_key {
            private_key.to_bytes().to_vec().into()
        } else {
            TCValue::None
        };

        TCValue::Vector(vec![
            actor.host.into(),
            actor.id,
            private_key,
            actor.public_key.to_bytes().to_vec().into(),
        ])
    }
}

impl TryFrom<TCValue> for Actor {
    type Error = error::TCError;

    fn try_from(value: TCValue) -> TCResult<Actor> {
        let mut value: Vec<TCValue> = value.try_into()?;
        if value.len() == 4 {
            let public_key: Bytes = value.pop().unwrap().try_into()?;

            Ok(Actor {
                public_key: PublicKey::from_bytes(&public_key[..])
                    .map_err(|_| error::unauthorized("Invalid public key specified for Actor"))?,
                private_key: if let Some(TCValue::Bytes(b)) = value.pop() {
                    Some(SecretKey::from_bytes(&b[..]).map_err(|_| {
                        error::unauthorized("Invalid private key specified for Actor")
                    })?)
                } else {
                    None
                },
                id: value.pop().unwrap(),
                host: value.pop().unwrap().try_into()?,
            })
        } else {
            let value: TCValue = value.into();
            Err(error::bad_request("Expected Actor, found", value))
        }
    }
}

#[async_trait]
impl TCObject for Actor {
    fn class() -> &'static str {
        "Actor"
    }

    fn id(&self) -> TCValue {
        self.id.clone()
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
