use std::collections::{BTreeSet, HashSet, VecDeque};
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use aes_gcm_siv::aead::rand_core::{OsRng, RngCore};
use aes_gcm_siv::aead::Aead;
use aes_gcm_siv::{Aes256GcmSiv, KeyInit, Nonce};
use async_trait::async_trait;
use log::{debug, trace, warn};
use safecast::{TryCastFrom, TryCastInto};
use umask::Mode;

use tc_error::*;
use tc_scalar::OpRefType;
use tc_state::CacheBlock;
use tc_transact::public::*;
use tc_transact::{fs, Gateway, Transact, Transaction, TxnId};
use tc_value::{Host, Link, TCString, Value};
use tcgeneric::{label, Label, Map, NetworkTime, PathSegment, TCPath, TCPathBuf};

use crate::cluster::{Class, Cluster, Dir, DirEntry, ReplicateAndJoin};
use crate::txn::{Hypothetical, Txn, TxnServer};
use crate::{aes256, cluster, Authorize, SignedToken, State};

const CLASS: Label = label("class");
const REPLICATION_TTL: Duration = Duration::from_secs(30);

pub struct Endpoint<'a> {
    mode: Mode,
    txn: &'a Txn,
    path: &'a [PathSegment],
    handler: Box<dyn Handler<'a, State> + 'a>,
}

impl<'a> Endpoint<'a> {
    pub fn umask(&self) -> Mode {
        self.mode
    }

    pub fn get(self, key: Value) -> TCResult<GetFuture<'a, State>> {
        let get = self
            .handler
            .get()
            .ok_or_else(|| TCError::method_not_allowed(OpRefType::Get, TCPath::from(self.path)))?;

        if self.mode.may_read() {
            Ok((get)(self.txn, key))
        } else {
            Err(unauthorized!("read {}", TCPath::from(self.path)))
        }
    }

    pub fn put(self, key: Value, value: State) -> TCResult<PutFuture<'a>> {
        let put = self
            .handler
            .put()
            .ok_or_else(|| TCError::method_not_allowed(OpRefType::Put, TCPath::from(self.path)))?;

        if self.mode.may_write() {
            Ok((put)(self.txn, key, value))
        } else {
            Err(unauthorized!("read {}", TCPath::from(self.path)))
        }
    }

    pub fn post(self, params: Map<State>) -> TCResult<PostFuture<'a, State>> {
        let post = self
            .handler
            .post()
            .ok_or_else(|| TCError::method_not_allowed(OpRefType::Post, TCPath::from(self.path)))?;

        if self.mode.may_execute() {
            Ok((post)(self.txn, params))
        } else {
            Err(unauthorized!("execute {}", TCPath::from(self.path)))
        }
    }

    pub fn delete(self, key: Value) -> TCResult<DeleteFuture<'a>> {
        let delete = self.handler.delete().ok_or_else(|| {
            TCError::method_not_allowed(OpRefType::Delete, TCPath::from(self.path))
        })?;

        if self.mode.may_write() {
            Ok((delete)(self.txn, key))
        } else {
            Err(unauthorized!("read {}", TCPath::from(self.path)))
        }
    }
}

pub(crate) struct Kernel {
    class: Cluster<Dir<Class>>,
    hypothetical: Cluster<Hypothetical>,
    keys: HashSet<aes256::Key>,
}

impl Kernel {
    fn issue_token(&self, txn_id: TxnId, path: &[PathSegment]) -> TCResult<SignedToken> {
        if path.is_empty() {
            Err(bad_request!(
                "cannot issue a token for {}",
                TCPath::from(path)
            ))
        } else if path[0] == CLASS {
            issue_token(txn_id, &self.class, path)
        } else if path.len() >= 2 && &path[..2] == &Hypothetical::PATH[..] {
            Err(bad_request!(
                "cannot issue a token for {}",
                TCPath::from(path)
            ))
        } else {
            Err(not_found!("{}", TCPath::from(path)))
        }
    }

    pub fn authorize_claim_and_route<'a>(
        &'a self,
        path: &'a [PathSegment],
        txn: &'a Txn,
    ) -> TCResult<Endpoint<'a>> {
        if path.is_empty() {
            Ok(Endpoint {
                mode: Mode::new().with_class_perm(umask::OTHERS, umask::READ),
                txn,
                path,
                handler: Box::new(KernelHandler::from(self)),
            })
        } else if path[0] == CLASS {
            auth_claim_route(&self.class, &path[1..], txn)
        } else if path.len() >= 2 && &path[..2] == &Hypothetical::PATH[..] {
            auth_claim_route(&self.hypothetical, &path[2..], txn)
        } else {
            Err(not_found!("{}", TCPath::from(path)))
        }
    }

    pub async fn replicate_and_join(
        &self,
        txn_server: TxnServer,
        peers: BTreeSet<Host>,
    ) -> Result<(), bool> {
        let mut progress = false;
        let mut unvisited = VecDeque::new();
        unvisited.push_back(Box::new(self.class.clone()));

        while let Some(cluster) = unvisited.pop_front() {
            let mut joined = false;

            for peer in &peers {
                let txn = txn_server.create_txn(NetworkTime::now());
                let txn_id = *txn.id();
                let txn = match get_token(&txn, peer, &self.keys, cluster.path()).await {
                    Ok(token) => match txn_server
                        .verify_txn(txn_id, NetworkTime::now(), token)
                        .await
                    {
                        Ok(txn) => txn,
                        Err(cause) => {
                            warn!("failed to verify token from {peer}: {cause}");
                            continue;
                        }
                    },
                    Err(cause) => {
                        warn!("failed to fetch token from {peer}: {cause}");
                        continue;
                    }
                };

                match cluster.replicate_and_join(&txn, peer.clone()).await {
                    Ok(entries) => {
                        joined = true;
                        progress = true;

                        for (_name, entry) in entries {
                            match &*entry {
                                DirEntry::Dir(dir) => unvisited.push_back(Box::new(dir.clone())),
                                DirEntry::Item(_) => {}
                            }
                        }
                    }
                    Err(cause) => warn!("failed to replicate from {peer}: {cause}"),
                }
            }

            if !joined {
                return Err(progress);
            }
        }

        // TODO: replicate services in the /services dir

        Ok(())
    }
}

impl Kernel {
    pub async fn commit(&self, txn_id: TxnId) {
        self.hypothetical.rollback(&txn_id).await;
    }

    pub async fn finalize(&self, txn_id: TxnId) {
        self.hypothetical.finalize(&txn_id).await;
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct Schema {
    host: Host,
    owner: Option<Link>,
    group: Option<Link>,
    keys: HashSet<aes256::Key>,
}

impl Schema {
    pub fn new(
        host: Host,
        owner: Option<Link>,
        group: Option<Link>,
        keys: HashSet<aes256::Key>,
    ) -> Self {
        Self {
            host,
            owner,
            group,
            keys,
        }
    }
}

#[async_trait]
impl fs::Persist<CacheBlock> for Kernel {
    type Txn = Txn;
    type Schema = Schema;

    async fn create(
        txn_id: TxnId,
        schema: Self::Schema,
        store: fs::Dir<CacheBlock>,
    ) -> TCResult<Self> {
        let class_dir: fs::Dir<CacheBlock> = store.create_dir(txn_id, CLASS.into()).await?;

        let link = Link::new(schema.host.clone(), CLASS.into());
        let class_schema = cluster::Schema::new(link, schema.owner.clone(), schema.group.clone());
        let class = fs::Persist::<CacheBlock>::create(txn_id, class_schema, class_dir).await?;

        let link = Link::new(schema.host, Hypothetical::PATH.into());
        let txn_schema = cluster::Schema::new(link, schema.owner, schema.group);
        let hypothetical = Cluster::new(txn_schema, Hypothetical::new(), txn_id);

        Ok(Self {
            class,
            hypothetical,
            keys: schema.keys,
        })
    }

    async fn load(
        txn_id: TxnId,
        schema: Self::Schema,
        store: fs::Dir<CacheBlock>,
    ) -> TCResult<Self> {
        let class_dir: fs::Dir<CacheBlock> = store.get_or_create_dir(txn_id, CLASS.into()).await?;

        let link = Link::new(schema.host.clone(), CLASS.into());
        let class_schema = cluster::Schema::new(link, schema.owner.clone(), schema.group.clone());
        let class = fs::Persist::<CacheBlock>::load(txn_id, class_schema, class_dir).await?;

        let link = Link::new(schema.host, Hypothetical::PATH.into());
        let txn_schema = cluster::Schema::new(link, schema.owner, schema.group);
        let hypothetical = Cluster::new(txn_schema, Hypothetical::new(), txn_id);

        Ok(Self {
            class,
            hypothetical,
            keys: schema.keys,
        })
    }

    fn dir(&self) -> fs::Inner<CacheBlock> {
        unimplemented!("Kernel::inner")
    }
}

fn auth_claim_route<'a, T>(
    cluster: &'a Cluster<T>,
    path: &'a [PathSegment],
    txn: &'a Txn,
) -> TCResult<Endpoint<'a>>
where
    Cluster<T>: Route<State>,
{
    let txn_id = *txn.id();
    let keyring = cluster.keyring(txn_id)?;
    let resource_mode = cluster.umask(txn_id, path);
    let request_mode = txn.mode(keyring, path);
    let mode = resource_mode & request_mode;

    if mode == Mode::new() {
        return Err(unauthorized!("access to {}", TCPath::from(path)));
    } else {
        debug!("request permissions are {mode}");
    }

    let handler = cluster
        .route(&*path)
        .ok_or_else(|| TCError::not_found(TCPath::from(path)))?;

    let endpoint = Endpoint {
        mode,
        txn,
        path,
        handler,
    };

    Ok(endpoint)
}

async fn get_token(
    txn: &Txn,
    peer: &Host,
    keys: &HashSet<aes256::Key>,
    path: &TCPathBuf,
) -> TCResult<String> {
    let mut nonce = [0u8; 96];
    OsRng.fill_bytes(&mut nonce);

    let link = Link::from((peer.clone(), path.clone()));

    for key in keys {
        let cipher = Aes256GcmSiv::new(key);
        let path_encrypted = encrypt_path(&cipher, &nonce, path)?;

        let nonce_and_path = (
            Value::Bytes(nonce.into()),
            Value::Bytes(path_encrypted.into()),
        );

        let nonce_and_token = txn.get(link.clone(), nonce_and_path).await?;

        let (nonce, token): (Value, Value) =
            nonce_and_token.try_cast_into(|t| TCError::unexpected(t, "an encrypted auth token"))?;

        let nonce: Arc<[u8]> = nonce.try_into()?;
        let token: Arc<[u8]> = token.try_into()?;

        return decrypt_token(&cipher, &nonce, &token);
    }

    Err(internal!(
        "no peers provided an auth token for cluster replication"
    ))
}

fn issue_token<T: Clone + fmt::Debug>(
    txn_id: TxnId,
    cluster: &Cluster<Dir<T>>,
    path: &[PathSegment],
) -> TCResult<SignedToken> {
    match cluster.lookup(txn_id, path)? {
        (suffix, entry) if suffix.is_empty() => match entry {
            DirEntry::Dir(dir) => dir.issue_token(Mode::all(), REPLICATION_TTL),
            DirEntry::Item(item) => item.issue_token(Mode::all(), REPLICATION_TTL),
        },
        (suffix, _) => Err(not_found!("cluster at {}", TCPath::from(suffix))),
    }
}

struct KernelHandler<'a> {
    kernel: &'a Kernel,
}

impl<'a> Handler<'a, State> for KernelHandler<'a> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                let (nonce, path_encrypted): (Value, Value) =
                    key.try_cast_into(|v| TCError::unexpected(v, "an encrypted path"))?;

                let nonce: Arc<[u8]> =
                    nonce.try_cast_into(|v| TCError::unexpected(v, "nonce bytes"))?;

                let path_encrypted = TCString::try_cast_from(path_encrypted, |v| {
                    TCError::unexpected(v, "an encrypted path")
                })?;

                for key in &self.kernel.keys {
                    let cipher = Aes256GcmSiv::new(key);
                    match decrypt_path(&cipher, &nonce, path_encrypted.as_str().as_bytes()) {
                        Ok(path) => {
                            let mut nonce = [0u8; 96];
                            OsRng.fill_bytes(&mut nonce);

                            let signed_token = self.kernel.issue_token(*txn.id(), &path)?;
                            let token = signed_token.into_jwt();
                            let encrypted_token = encrypt_token(&cipher, &nonce, token)?;

                            return Ok(State::from(Value::Tuple(
                                vec![nonce.into(), encrypted_token.into()].into(),
                            )));
                        }
                        Err(cause) => {
                            trace!("unable to decrypt the requested cluster path: {cause}")
                        }
                    }
                }

                Err(bad_request!("unable to decrypt the requested cluster path"))
            })
        }))
    }
}

impl<'a> From<&'a Kernel> for KernelHandler<'a> {
    fn from(kernel: &'a Kernel) -> Self {
        Self { kernel }
    }
}

#[inline]
fn decrypt_path(cipher: &Aes256GcmSiv, nonce: &[u8], path_encrypted: &[u8]) -> TCResult<TCPathBuf> {
    let nonce = Nonce::from_slice(&nonce);
    match cipher.decrypt(nonce, path_encrypted) {
        Ok(path_decrypted) => {
            let path_decrypted = String::from_utf8(path_decrypted)
                .map_err(|cause| bad_request!("invalid UTF8: {cause}"))?;

            path_decrypted.parse().map_err(TCError::from)
        }
        Err(_cause) => Err(bad_request!("unable to decrypt the requested cluster path")),
    }
}

#[inline]
fn decrypt_token(cipher: &Aes256GcmSiv, nonce: &[u8], token_encrypted: &[u8]) -> TCResult<String> {
    let nonce = Nonce::from_slice(&nonce);
    match cipher.decrypt(nonce, token_encrypted) {
        Ok(token_decrypted) => String::from_utf8(token_decrypted)
            .map_err(|cause| bad_request!("invalid UTF8: {cause}")),

        Err(_cause) => Err(bad_request!("unable to decrypt the provided auth token")),
    }
}

#[inline]
fn encrypt_path(cipher: &Aes256GcmSiv, nonce: &[u8], path: &TCPathBuf) -> TCResult<Arc<[u8]>> {
    let path = path.to_string();
    let nonce = Nonce::from_slice(&nonce);

    cipher
        .encrypt(&nonce, path.as_bytes())
        .map(Arc::from)
        .map_err(|_| internal!("unable to encrypt path"))
}

#[inline]
fn encrypt_token(cipher: &Aes256GcmSiv, nonce: &[u8], token: String) -> TCResult<Arc<[u8]>> {
    let nonce = Nonce::from_slice(&nonce);

    cipher
        .encrypt(&nonce, token.as_bytes())
        .map(Arc::from)
        .map_err(|_| internal!("unable to encrypt token"))
}
