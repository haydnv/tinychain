use std::convert::{TryFrom, TryInto};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use ed25519_dalek::{Keypair, PublicKey, SecretKey, Signature};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};

use crate::error;
use crate::host::NetworkTime;
use crate::object::TCObject;
use crate::state::State;
use crate::transaction::Transaction;
use crate::value::{Args, Link, PathSegment, TCPath, TCResult, TCValue};

const TOKEN_DURATION: Duration = Duration::from_secs(30 * 60);

pub struct Actor {
    id: TCValue,
    public_key: PublicKey,
    private_key: Option<SecretKey>,
}

#[async_trait]
impl TCObject for Actor {
    async fn new(_txn: Arc<Transaction>, id: TCValue) -> TCResult<Arc<Actor>> {
        let mut rng = OsRng {};
        let keypair: Keypair = Keypair::generate(&mut rng);

        println!("OK: Actor::new");
        Ok(Arc::new(Actor {
            id,
            public_key: keypair.public,
            private_key: Some(keypair.secret),
        }))
    }

    fn class() -> &'static str {
        "Actor"
    }

    fn id(&self) -> TCValue {
        self.id.clone()
    }

    async fn post(
        &self,
        txn: Arc<Transaction>,
        method: &PathSegment,
        mut args: Args,
    ) -> TCResult<State> {
        match method.as_str() {
            "token" => {
                let issuer: Link = args.take("issuer")?;
                let scopes: Vec<TCPath> = args.take("scopes")?;
                let issued_at = txn.time();
                let expires = issued_at.clone() + TOKEN_DURATION;
                let token = self.token(issuer, scopes, issued_at, expires)?;
                Ok(token.into())
            }
            _ => Err(error::bad_request("Actor has no such method", method)),
        }
    }
}

impl Actor {
    pub fn token(
        &self,
        issuer: Link,
        scopes: Vec<TCPath>,
        issued_at: NetworkTime,
        expires: NetworkTime,
    ) -> TCResult<String> {
        let keypair = if let Some(secret) = &self.private_key {
            Keypair::from_bytes(&[secret.to_bytes(), self.public_key.to_bytes()].concat()).map_err(
                |_| error::unauthorized("Unable to construct ECDSA keypair for the given user"),
            )?
        } else {
            return Err(error::forbidden(
                "You are not authorized to issue tokens on behalf of this user".into(),
            ));
        };

        let header = TokenHeader::default();

        let claims = TokenClaims {
            iss: issuer,
            iat: issued_at.as_millis(),
            exp: expires.as_millis(),
            actor_id: self.id.clone(),
            scopes,
        };

        let header = base64::encode(serde_json::to_string_pretty(&header).unwrap());
        let claims = base64::encode(serde_json::to_string_pretty(&claims).unwrap());
        let signature = base64::encode(
            &keypair
                .sign(format!("{}.{}", header, claims).as_bytes())
                .to_bytes()[..],
        );
        Ok(format!("{}.{}.{}", header, claims, signature))
    }

    pub fn verify(&self, token: &str, scope: &TCPath) -> TCResult<()> {
        let err1 = |e| error::unauthorized(&format!("Invalid bearer token provided: {}", e));
        let err2 = |e| error::unauthorized(&format!("Invalid bearer token provided: {}", e));
        let mut token: Vec<&str> = token.split('.').collect();
        if token.len() != 3 {
            return Err(error::unauthorized(
                "Expected bearer token in the format '<header>.<claims>.<data>'",
            ));
        }

        let message = format!("{}.{}", token[0], token[1]);
        let signature = Signature::from_bytes(&base64::decode(token.pop().unwrap()).map_err(err1)?)
            .map_err(|_| error::unauthorized("Invalid bearer token signature"))?;

        let claims = token.pop().unwrap();
        let claims = base64::decode(claims).map_err(err1)?;
        let claims: TokenClaims = serde_json::from_slice(&claims).map_err(err2)?;

        if claims.actor_id != self.id {
            return Err(error::unauthorized(
                "Attempted to use a bearer token for a different user",
            ));
        }

        let header = token.pop().unwrap();
        let header = base64::decode(header).map_err(err1)?;
        let header: TokenHeader = serde_json::from_slice(&header).map_err(err2)?;

        if header != TokenHeader::default() {
            Err(error::unauthorized("Unsupported bearer token"))
        } else if self
            .public_key
            .verify(message.as_bytes(), &signature)
            .is_err()
        {
            Err(error::unauthorized("Invalid bearer token provided"))
        } else {
            for authorized_scope in claims.scopes {
                if scope.starts_with(authorized_scope) {
                    return Ok(());
                }
            }

            Err(error::forbidden(format!(
                "Not authorized for scope {}",
                scope
            )))
        }
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
        if value.len() == 3 {
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
            })
        } else {
            let value: TCValue = value.into();
            Err(error::bad_request("Expected Actor, found", value))
        }
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

#[derive(Deserialize, Serialize)]
struct TokenClaims {
    iss: Link,
    iat: u64,
    exp: u64,
    actor_id: TCValue,
    scopes: Vec<TCPath>,
}
