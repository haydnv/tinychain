use std::fmt;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use ed25519_dalek::{Keypair, PublicKey, SecretKey, Signature as ECSignature};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use signature::{Signature, Signer, Verifier};

use error::*;
use generic::{path_label, NetworkTime, PathLabel, TCPathBuf};
use value::{Link, Value};

pub type Scope = TCPathBuf;

pub const SCOPE_WRITE: PathLabel = path_label(&["write"]);
pub const SCOPE_READ: PathLabel = path_label(&["write", "read"]);
pub const SCOPE_EXECUTE: PathLabel = path_label(&["write", "read", "execute"]);

const NANO: u128 = 1_000_000_000;

#[derive(Clone, Deserialize, Serialize)]
pub struct Token {
    iss: Link,
    iat: u64,
    exp: u64,
    actor_id: Value,
    scopes: Vec<Scope>,
}

impl Token {
    pub fn new(iss: Link, iat: NetworkTime, ttl: Duration, actor_id: Value, scopes: Vec<Scope>) -> Self {
        let iat = (iat.as_nanos() / NANO) as u64;
        let exp = iat + (ttl.as_nanos() / NANO) as u64;

        Self {
            iss,
            iat,
            exp,
            actor_id,
            scopes,
        }
    }

    pub fn actor_id(&self) -> (Link, Value) {
        (self.iss.clone(), self.actor_id.clone())
    }

    pub fn contains(&self, other: &Self) -> bool {
        for other_scope in &other.scopes {
            let mut valid = false;
            for scope in &self.scopes {
                if other_scope.starts_with(scope) {
                    valid = true;
                }
            }

            if !valid {
                return false;
            }
        }

        true
    }

    pub fn validate<I: fmt::Display>(&self, scope: Scope, id: I) -> TCResult<()> {
        for authorized in &self.scopes {
            if scope.starts_with(&authorized) {
                return Ok(());
            }
        }

        Err(TCError::forbidden(
            format!("The requested action requires the {} scope", scope),
            id,
        ))
    }
}

impl FromStr for Token {
    type Err = TCError;

    fn from_str(token: &str) -> TCResult<Token> {
        let token: Vec<&str> = token.split('.').collect();
        if token.len() != 3 {
            return Err(TCError::unauthorized(
                "Expected bearer token in the format '<header>.<claims>.<data>'",
            ));
        }

        let token = base64::decode(token[1])
            .map_err(|e| TCError::unauthorized(&format!("Invalid bearer token: {}", e)))?;

        let token: Token = serde_json::from_slice(&token)
            .map_err(|e| TCError::unauthorized(&format!("Invalid bearer token: {}", e)))?;

        Ok(token)
    }
}

pub struct Actor {
    host: Link,
    id: Value,
    public_key: PublicKey,
    private_key: SecretKey,
}

impl Actor {
    pub fn new(host: Link, id: Value) -> Arc<Actor> {
        let mut rng = OsRng {};
        let keypair: Keypair = Keypair::generate(&mut rng);

        Arc::new(Actor {
            host,
            id,
            public_key: keypair.public,
            private_key: keypair.secret,
        })
    }

    pub fn sign_token(&self, token: &Token) -> TCResult<String> {
        let keypair = Keypair::from_bytes(
            &[self.private_key.to_bytes(), self.public_key.to_bytes()].concat(),
        )
        .map_err(|cause| {
            TCError::unauthorized(format!("Unable to construct ECDSA keypair: {}", cause))
        })?;

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
            return Err(TCError::unauthorized(
                "Expected bearer token in the format '<header>.<claims>.<data>'",
            ));
        }

        let message = format!("{}.{}", encoded[0], encoded[1]);
        let signature =
            ECSignature::from_bytes(&base64::decode(encoded.pop().unwrap()).map_err(|e| {
                TCError::unauthorized(&format!("Invalid bearer token signature: {}", e))
            })?)
            .map_err(|_| TCError::unauthorized("Invalid bearer token signature"))?;

        let token = encoded.pop().unwrap();
        let token = base64::decode(token)
            .map_err(|e| TCError::unauthorized(&format!("Invalid bearer token: {}", e)))?;
        let token: Token = serde_json::from_slice(&token)
            .map_err(|e| TCError::unauthorized(&format!("Invalid bearer token: {}", e)))?;

        if token.actor_id != self.id {
            return Err(TCError::unauthorized(
                "Attempted to use a bearer token for a different actor",
            ));
        }

        let header = encoded.pop().unwrap();
        let header = base64::decode(header)
            .map_err(|e| TCError::unauthorized(&format!("Invalid bearer token header: {}", e)))?;
        let header: TokenHeader = serde_json::from_slice(&header)
            .map_err(|e| TCError::unauthorized(&format!("Invalid bearer token header: {}", e)))?;

        if header != TokenHeader::default() {
            Err(TCError::unauthorized("Unsupported bearer token type"))
        } else if self
            .public_key
            .verify(message.as_bytes(), &signature)
            .is_err()
        {
            Err(TCError::unauthorized("Invalid bearer token provided"))
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

#[derive(Eq, PartialEq, Deserialize, Serialize)]
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
