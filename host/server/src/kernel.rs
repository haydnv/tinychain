use std::collections::{BTreeSet, HashSet, VecDeque};
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use aes_gcm_siv::aead::rand_core::{OsRng, RngCore};
use aes_gcm_siv::aead::Aead;
use aes_gcm_siv::{Aes256GcmSiv, KeyInit};
use async_trait::async_trait;
use futures::join;
use log::{debug, info, trace, warn};
use rand::prelude::IteratorRandom;
use rjwt::VerifyingKey;
use safecast::TryCastInto;
use umask::Mode;

use tc_error::*;
use tc_scalar::OpRefType;
#[cfg(feature = "service")]
use tc_state::chain::Recover;
use tc_state::CacheBlock;
use tc_transact::hash::AsyncHash;
use tc_transact::public::*;
use tc_transact::{fs, Gateway, Replicate, Transact, Transaction, TxnId};
use tc_value::{Host, Link, ToUrl, Value};
use tcgeneric::{label, Label, Map, NetworkTime, PathSegment, TCPath, TCPathBuf, Tuple};

use crate::client::Egress;
#[cfg(feature = "service")]
use crate::cluster::Service;
use crate::cluster::{Class, Cluster, Dir, DirEntry, IsDir, Library, ReplicateAndJoin};
use crate::txn::{Hypothetical, Txn, TxnServer};
use crate::{aes256, cluster, Authorize, SignedToken, State};

pub const CLASS: Label = label("class");
pub const LIB: Label = label("lib");
pub const SERVICE: Label = label("service");
const REPLICATION_TTL: Duration = Duration::from_secs(30);
const STATE_MODE: Mode = Mode::new()
    .with_class_perm(umask::OTHERS, umask::READ)
    .with_class_perm(umask::OTHERS, umask::EXEC);

#[cfg(not(feature = "service"))]
const ERR_NOT_ENABLED: &str = "this binary was compiled without the 'service' feature";

type Nonce = [u8; 12];

struct KernelEgress;

impl Egress for KernelEgress {
    fn is_authorized(&self, _link: &ToUrl<'_>, _write: bool) -> bool {
        // TODO: implement authorization logic
        true
    }
}

impl fmt::Debug for KernelEgress {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("kernel egress policy")
    }
}

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
    library: Cluster<Dir<Library>>,
    #[cfg(feature = "service")]
    service: Cluster<Dir<Service>>,
    state: tc_state::public::Static<Txn>,
    hypothetical: Cluster<Hypothetical>,
    keys: HashSet<aes256::Key>,
}

impl Kernel {
    async fn issue_token(&self, txn_id: TxnId, path: &[PathSegment]) -> TCResult<SignedToken> {
        if path.is_empty() {
            Err(bad_request!(
                "cannot issue a token for {}",
                TCPath::from(path)
            ))
        } else if path[0] == SERVICE {
            #[cfg(feature = "service")]
            {
                issue_token(txn_id, self.service.clone(), &path[1..]).await
            }

            #[cfg(not(feature = "service"))]
            {
                Err(not_implemented!("{ERR_NOT_ENABLED}"))
            }
        } else if path[0] == LIB {
            issue_token(txn_id, self.library.clone(), &path[1..]).await
        } else if path[0] == CLASS {
            issue_token(txn_id, self.class.clone(), &path[1..]).await
        } else if path.len() >= 2 && &path[..2] == &Hypothetical::PATH[..] {
            Err(bad_request!(
                "cannot issue a token for {}",
                TCPath::from(path)
            ))
        } else {
            Err(not_found!(
                "there is no resource at {} to issue a token",
                TCPath::from(path)
            ))
        }
    }

    pub async fn public_key(&self, txn_id: TxnId, path: &[PathSegment]) -> TCResult<VerifyingKey> {
        if path.is_empty() {
            Err(bad_request!("{} has no public key", TCPath::from(path)))
        } else if path[0] == SERVICE {
            #[cfg(feature = "service")]
            {
                public_key(txn_id, self.service.clone(), &path[1..]).await
            }

            #[cfg(not(feature = "service"))]
            {
                Err(not_implemented!("{ERR_NOT_ENABLED}"))
            }
        } else if path[0] == LIB {
            public_key(txn_id, self.library.clone(), &path[1..]).await
        } else if path[0] == CLASS {
            public_key(txn_id, self.class.clone(), &path[1..]).await
        } else if path.len() >= 2 && &path[..2] == &Hypothetical::PATH[..] {
            if path.len() == Hypothetical::PATH.len() {
                Err(bad_request!("{} has no public key", TCPath::from(path)))
            } else {
                Err(not_found!("there is no resource at {}", TCPath::from(path)))
            }
        } else {
            Err(not_found!("there is no resource at {}", TCPath::from(path)))
        }
    }

    pub async fn route<'a>(
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
        } else if path[0] == State::PREFIX {
            let path = &path[1..];
            let handler = self
                .state
                .route(path)
                .ok_or_else(|| TCError::not_found(TCPath::from(path)))?;

            Ok(Endpoint {
                mode: STATE_MODE,
                txn,
                path,
                handler,
            })
        } else if path[0] == SERVICE {
            #[cfg(feature = "service")]
            {
                let (path, dir_entry) = self.service.clone().lookup(*txn.id(), &path[1..]).await?;

                match dir_entry {
                    DirEntry::Dir(cluster) => auth_claim_route(cluster, path, txn).await,
                    DirEntry::Item(cluster) => auth_claim_route(cluster, path, txn).await,
                }
            }

            #[cfg(not(feature = "service"))]
            {
                Err(not_implemented!("{ERR_NOT_ENABLED}"))
            }
        } else if path[0] == LIB {
            let (path, dir_entry) = self.library.clone().lookup(*txn.id(), &path[1..]).await?;

            match dir_entry {
                DirEntry::Dir(cluster) => auth_claim_route(cluster, path, txn).await,
                DirEntry::Item(cluster) => auth_claim_route(cluster, path, txn).await,
            }
        } else if path[0] == CLASS {
            let (path, dir_entry) = self.class.clone().lookup(*txn.id(), &path[1..]).await?;

            match dir_entry {
                DirEntry::Dir(cluster) => auth_claim_route(cluster, path, txn).await,
                DirEntry::Item(cluster) => auth_claim_route(cluster, path, txn).await,
            }
        } else if path.len() >= 2 && &path[..2] == &Hypothetical::PATH[..] {
            auth_claim_route(self.hypothetical.clone(), &path[2..], txn).await
        } else {
            Err(TCError::not_found(TCPath::from(path)))
        }
    }
    pub async fn replicate_and_join(
        &self,
        txn_server: &TxnServer,
        peers: &BTreeSet<Host>,
    ) -> Result<(), bool> {
        debug!("Kernel::replicate_and_join {peers:?}");

        if peers.is_empty() {
            info!("not joining replica set since no peers were provided");
            return Ok(());
        }

        Self::replicate_and_join_dir(&self.keys, &self.class, txn_server, peers).await?;
        Self::replicate_and_join_dir(&self.keys, &self.library, txn_server, peers).await?;
        #[cfg(feature = "service")]
        Self::replicate_and_join_dir(&self.keys, &self.service, txn_server, peers).await?;

        Self::replicate_and_join_items(&self.keys, &self.class, txn_server, peers).await?;
        Self::replicate_and_join_items(&self.keys, &self.library, txn_server, peers).await?;
        #[cfg(feature = "service")]
        Self::replicate_and_join_items(&self.keys, &self.service, txn_server, peers).await?;

        Ok(())
    }

    async fn replicate_and_join_dir<T>(
        keys: &HashSet<aes256::Key>,
        parent: &Cluster<Dir<T>>,
        txn_server: &TxnServer,
        peers: &BTreeSet<Host>,
    ) -> Result<(), bool>
    where
        T: Clone + fmt::Debug,
        Cluster<Dir<T>>: ReplicateAndJoin,
    {
        debug!(
            "Kernel::replicate_and_join_dir {} with peers {:?}",
            parent.path(),
            peers
        );

        let mut progress = false;
        let mut unvisited = VecDeque::new();
        unvisited.push_back(parent.clone());

        while let Some(cluster) = unvisited.pop_front() {
            let mut joined = false;

            for peer in peers.iter().choose_multiple(&mut OsRng, peers.len()) {
                let egress = Arc::new(KernelEgress);

                let txn = txn_server
                    .create_txn(NetworkTime::now())
                    .expect("txn")
                    .with_egress(egress.clone());

                trace!("fetching replication token...");

                let txn = match get_and_verify_token_from_peer(
                    &txn_server,
                    &txn,
                    peer,
                    keys,
                    cluster.path(),
                )
                .await
                {
                    Ok(txn) => txn.with_egress(egress.clone()),
                    Err(cause) => {
                        warn!("failed to fetch and verify token from {peer}: {cause}");
                        continue;
                    }
                };

                trace!("replicating {cluster:?}...");

                let txn_id = *txn.id();
                match cluster.replicate_and_join(txn, peer.clone()).await {
                    Ok(()) => {
                        joined = true;
                        progress = true;

                        let entries = cluster.entries(txn_id).await.expect("dir entry list");

                        for (_name, entry) in entries {
                            match &*entry {
                                DirEntry::Dir(dir) => unvisited.push_back(dir.clone()),
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

        Ok(())
    }

    async fn replicate_and_join_items<T>(
        keys: &HashSet<aes256::Key>,
        parent: &Cluster<Dir<T>>,
        txn_server: &TxnServer,
        peers: &BTreeSet<Host>,
    ) -> Result<(), bool>
    where
        T: Replicate<Txn> + Transact + Clone + fmt::Debug,
    {
        info!(
            "Kernel::replicate_and_join_dir {} with peers {:?}",
            parent.path(),
            peers
        );

        let mut progress = false;
        let mut unvisited = VecDeque::new();
        unvisited.push_back(DirEntry::Dir(parent.clone()));

        while let Some(cluster) = unvisited.pop_front() {
            let txn = txn_server.create_txn(NetworkTime::now()).expect("txn");
            let txn_id = *txn.id();

            let item = match cluster {
                DirEntry::Dir(dir) => {
                    let entries = dir.entries(txn_id).await.expect("dir entry list");
                    let entries = entries
                        .into_iter()
                        .map(|(_name, entry)| DirEntry::clone(&*entry));

                    unvisited.extend(entries);

                    continue;
                }
                DirEntry::Item(item) => item,
            };

            trace!("replicating {item:?}...");

            let mut joined = false;
            for peer in peers.iter().choose_multiple(&mut OsRng, peers.len()) {
                let egress = Arc::new(KernelEgress);
                let txn = txn.clone().with_egress(egress.clone());

                trace!("fetching replication token...");

                let txn = match get_and_verify_token_from_peer(
                    &txn_server,
                    &txn,
                    peer,
                    keys,
                    item.path(),
                )
                .await
                {
                    Ok(txn) => txn.with_egress(egress.clone()),
                    Err(cause) => {
                        warn!("failed to fetch and verify token from {peer}: {cause}");
                        continue;
                    }
                };

                info!("replicating {item:?} from {peer}...");

                match item.replicate_and_join(txn, peer.clone()).await {
                    Ok(()) => {
                        joined = true;
                        progress = true;
                    }
                    Err(cause) => warn!("failed to replicate from {peer}: {cause}"),
                }
            }

            if !joined {
                return Err(progress);
            }
        }

        Ok(())
    }
}

impl Kernel {
    pub async fn commit(&self, txn_id: TxnId) {
        debug!("Kernel::commit");

        join!(
            self.class.commit(txn_id),
            self.library.commit(txn_id),
            self.hypothetical.rollback(&txn_id)
        );
    }

    pub async fn finalize(&self, txn_id: &TxnId) {
        trace!("Kernel::finalize");

        join!(
            self.class.finalize(txn_id),
            self.library.finalize(txn_id),
            self.hypothetical.finalize(txn_id)
        );
    }
}

#[derive(Clone, Eq, PartialEq)]
pub struct Schema {
    lead: Host,
    host: Host,
    owner: Option<Link>,
    group: Option<Link>,
    keys: HashSet<aes256::Key>,
}

impl Schema {
    pub fn new(
        lead: Host,
        host: Host,
        owner: Option<Link>,
        group: Option<Link>,
        keys: HashSet<aes256::Key>,
    ) -> Self {
        Self {
            lead,
            host,
            owner,
            group,
            keys,
        }
    }

    fn to_cluster(&self, prefix: Label) -> cluster::Schema {
        cluster::Schema::new(
            self.lead.clone(),
            Link::new(self.host.clone(), prefix.into()),
            self.owner.clone(),
            self.group.clone(),
        )
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
        let lead = schema.lead.clone();

        let class = {
            let schema = schema.to_cluster(CLASS);
            let dir: fs::Dir<CacheBlock> = store.create_dir(txn_id, CLASS.into()).await?;
            fs::Persist::<CacheBlock>::create(txn_id, schema, dir).await?
        };

        let library = {
            let schema = schema.to_cluster(LIB);
            let dir: fs::Dir<CacheBlock> = store.create_dir(txn_id, LIB.into()).await?;
            fs::Persist::<CacheBlock>::create(txn_id, schema, dir).await?
        };

        #[cfg(feature = "service")]
        let service = {
            let schema = schema.to_cluster(SERVICE);
            let dir: fs::Dir<CacheBlock> = store.create_dir(txn_id, SERVICE.into()).await?;
            fs::Persist::<CacheBlock>::create(txn_id, schema, dir).await?
        };

        let link = Link::new(schema.host, Hypothetical::PATH.into());
        let txn_schema = cluster::Schema::new(lead, link, schema.owner, schema.group);
        let hypothetical = Cluster::new(txn_schema, Hypothetical::new());

        Ok(Self {
            class,
            library,
            hypothetical,
            #[cfg(feature = "service")]
            service,
            state: tc_state::public::Static::default(),
            keys: schema.keys,
        })
    }

    async fn load(
        txn_id: TxnId,
        schema: Self::Schema,
        store: fs::Dir<CacheBlock>,
    ) -> TCResult<Self> {
        let lead = schema.lead.clone();

        let class = {
            let schema = schema.to_cluster(CLASS);
            let dir = store.get_or_create_dir(txn_id, CLASS.into()).await?;
            fs::Persist::<CacheBlock>::load(txn_id, schema, dir).await?
        };

        let library = {
            let schema = schema.to_cluster(LIB);
            let dir = store.get_or_create_dir(txn_id, LIB.into()).await?;
            fs::Persist::<CacheBlock>::load(txn_id, schema, dir).await?
        };

        #[cfg(feature = "service")]
        let service = {
            let schema = schema.to_cluster(SERVICE);
            let dir = store.get_or_create_dir(txn_id, SERVICE.into()).await?;
            fs::Persist::<CacheBlock>::load(txn_id, schema, dir).await?
        };

        let link = Link::new(schema.host, Hypothetical::PATH.into());
        let txn_schema = cluster::Schema::new(lead, link, schema.owner, schema.group);
        let hypothetical = Cluster::new(txn_schema, Hypothetical::new());

        Ok(Self {
            class,
            library,
            #[cfg(feature = "service")]
            service,
            hypothetical,
            state: tc_state::public::Static::default(),
            keys: schema.keys,
        })
    }

    fn dir(&self) -> fs::Inner<CacheBlock> {
        unimplemented!("Kernel::inner")
    }
}

#[cfg(feature = "service")]
#[async_trait]
impl Recover<CacheBlock> for Kernel {
    type Txn = Txn;

    async fn recover(&self, txn: &Txn) -> TCResult<()> {
        self.service.recover(txn).await
    }
}

async fn auth_claim_route<'a, T>(
    cluster: Cluster<T>,
    path: &'a [PathSegment],
    txn: &'a Txn,
) -> TCResult<Endpoint<'a>>
where
    T: AsyncHash + Route<State> + IsDir + Transact + Send + Sync + fmt::Debug + 'a,
{
    let txn_id = *txn.id();
    let mode = {
        let resource_mode = cluster.umask(txn_id, path);
        let request_mode = if txn.has_claims() {
            let keyring = cluster.keyring(txn_id).await?;
            txn.mode(keyring, path)
        } else {
            Txn::DEFAULT_MODE
        };

        resource_mode & request_mode
    };

    if mode == Mode::new() {
        return Err(unauthorized!("access to {}", TCPath::from(path)));
    }

    let handler = cluster
        .route_owned(&*path)
        .ok_or_else(|| not_found!("endpoint {}", TCPath::from(path)))?;

    let endpoint = Endpoint {
        mode,
        txn,
        path,
        handler,
    };

    Ok(endpoint)
}

async fn get_and_verify_token_from_peer(
    txn_server: &TxnServer,
    txn: &Txn,
    peer: &Host,
    keys: &HashSet<aes256::Key>,
    path: &TCPathBuf,
) -> TCResult<Txn> {
    let token = get_token_from_peer(&txn, peer, keys, path).await?;

    txn_server
        .verify_txn(*txn.id(), NetworkTime::now(), token)
        .await
}

async fn get_token_from_peer(
    txn: &Txn,
    peer: &Host,
    keys: &HashSet<aes256::Key>,
    path: &TCPathBuf,
) -> TCResult<String> {
    let mut nonce = [0u8; 12];
    OsRng.fill_bytes(&mut nonce);

    let link = Link::from(peer.clone());

    for key in keys {
        let cipher = Aes256GcmSiv::new(key);
        let path_encrypted = encrypt_path(&cipher, &nonce, path)?;

        let nonce_and_path = (
            Value::Bytes(nonce.into()),
            Value::Bytes(path_encrypted.into()),
        );

        trace!("requesting replication token from {link}...");

        let nonce_and_token = txn.get(link.clone(), nonce_and_path).await?;

        let (nonce, token): (Value, Value) =
            Tuple::<State>::try_from(nonce_and_token).and_then(|tuple| {
                tuple.try_cast_into(|t| TCError::unexpected(t, "an encrypted auth token"))
            })?;

        let nonce: Arc<[u8]> = nonce.try_into()?;
        let token: Arc<[u8]> = token.try_into()?;

        return decrypt_token(&cipher, &nonce, &token);
    }

    Err(internal!(
        "no peers provided an auth token for cluster replication"
    ))
}

async fn issue_token<T>(
    txn_id: TxnId,
    cluster: Cluster<Dir<T>>,
    path: &[PathSegment],
) -> TCResult<SignedToken>
where
    T: Clone + Send + Sync + fmt::Debug,
{
    match cluster.lookup(txn_id, path).await? {
        (suffix, entry) if suffix.is_empty() => match entry {
            DirEntry::Dir(dir) => dir.issue_token(Mode::all(), REPLICATION_TTL),
            DirEntry::Item(item) => item.issue_token(Mode::all(), REPLICATION_TTL),
        },
        (suffix, _) => Err(not_found!("cluster at {}", TCPath::from(suffix))),
    }
}

async fn public_key<T>(
    txn_id: TxnId,
    cluster: Cluster<Dir<T>>,
    path: &[PathSegment],
) -> TCResult<VerifyingKey>
where
    T: Clone + Send + Sync + fmt::Debug,
{
    match cluster.lookup(txn_id, path).await? {
        (suffix, entry) if suffix.is_empty() => match entry {
            DirEntry::Dir(dir) => Ok(dir.public_key()),
            DirEntry::Item(item) => Ok(item.public_key()),
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
                trace!("GET /?key={key:?}");

                let (nonce, path_encrypted): (Value, Value) =
                    key.try_cast_into(|v| TCError::unexpected(v, "an encrypted path"))?;

                let nonce: Arc<[u8]> =
                    nonce.try_cast_into(|v| TCError::unexpected(v, "nonce bytes"))?;

                let path_encrypted: Arc<[u8]> = path_encrypted
                    .try_cast_into(|v| TCError::unexpected(v, "an encrypted path"))?;

                for key in &self.kernel.keys {
                    let cipher = Aes256GcmSiv::new(key);

                    match decrypt_path(&cipher, &nonce, &*path_encrypted) {
                        Ok(path) => {
                            let mut nonce = [0u8; 12];
                            OsRng.fill_bytes(&mut nonce);

                            let signed_token = self.kernel.issue_token(*txn.id(), &path).await?;
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
    let nonce = Nonce::try_from(nonce).map_err(|cause| bad_request!("invalid nonce: {cause}"))?;

    match cipher.decrypt(&nonce.into(), path_encrypted) {
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
    let nonce = Nonce::try_from(nonce).map_err(|cause| bad_request!("invalid nonce: {cause}"))?;

    match cipher.decrypt(&nonce.into(), token_encrypted) {
        Ok(token_decrypted) => String::from_utf8(token_decrypted)
            .map_err(|cause| bad_request!("invalid UTF8: {cause}")),

        Err(_cause) => Err(bad_request!("unable to decrypt the provided auth token")),
    }
}

#[inline]
fn encrypt_path(cipher: &Aes256GcmSiv, nonce: &[u8], path: &TCPathBuf) -> TCResult<Arc<[u8]>> {
    let path = path.to_string();
    let nonce = Nonce::try_from(nonce).map_err(|cause| bad_request!("invalid nonce: {cause}"))?;

    cipher
        .encrypt(&nonce.into(), path.as_bytes())
        .map(Arc::from)
        .map_err(|_| internal!("unable to encrypt path"))
}

#[inline]
fn encrypt_token(cipher: &Aes256GcmSiv, nonce: &[u8], token: String) -> TCResult<Arc<[u8]>> {
    let nonce = Nonce::try_from(nonce).map_err(|cause| bad_request!("invalid nonce: {cause}"))?;

    cipher
        .encrypt(&nonce.into(), token.as_bytes())
        .map(Arc::from)
        .map_err(|_| internal!("unable to encrypt token"))
}
