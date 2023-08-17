use std::fmt;

use async_trait::async_trait;
use log::debug;
use safecast::{TryCastFrom, TryCastInto};

use tc_chain::{ChainInstance, CHAIN};
use tc_error::*;
use tc_fs::CacheBlock;
use tc_scalar::Scalar;
use tc_state::chain::{BlockChain, Chain, SyncChain};
use tc_state::collection::CollectionBase;
use tc_state::object::InstanceClass;
use tc_state::State;
use tc_transact::fs::{Persist, Restore};
use tc_transact::public::{Public, Route, ToState};
use tc_transact::{RPCClient, Transact, Transaction, TxnId};
use tc_value::{Link, Value, Version as VersionNumber};
use tcgeneric::{label, Label, Map};

use crate::txn::Txn;

/// The name of the endpoint which serves a [`Link`] to each of this [`Cluster`]'s replicas.
pub const REPLICAS: Label = label("replicas");

/// A state which supports replication in a [`Cluster`]
#[async_trait]
pub trait Replica {
    async fn state(&self, txn_id: TxnId) -> TCResult<State>;

    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()>;
}

#[async_trait]
impl Replica for BlockChain<crate::cluster::Class> {
    async fn state(&self, txn_id: TxnId) -> TCResult<State> {
        self.subject().to_state(txn_id).await
    }

    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()> {
        let params = Map::one(label("add"), txn.host().clone().into());

        let state = txn
            .post(source.append(REPLICAS), State::Map(params))
            .await?;

        let classes: Map<Map<InstanceClass>> =
            state.try_cast_into(|s| TCError::unexpected(s, "Class version history"))?;

        // TODO: verify equality of existing versions
        let latest_version = self.subject().latest(*txn.id()).await?;
        for (number, version) in classes {
            let number: VersionNumber = number.as_str().parse()?;
            if let Some(latest) = latest_version {
                if number > latest {
                    self.put(txn, &[], number.into(), version.into()).await?;
                }
            } else {
                self.put(txn, &[], number.into(), version.into()).await?;
            }
        }

        Ok(())
    }
}

#[async_trait]
impl Replica for BlockChain<crate::cluster::Library> {
    async fn state(&self, txn_id: TxnId) -> TCResult<State> {
        self.subject().to_state(txn_id).await
    }

    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()> {
        let params = Map::one(label("add"), txn.host().clone().into());

        let state = txn
            .post(source.clone().append(REPLICAS), State::Map(params))
            .await?;

        let library: Map<Map<Scalar>> =
            state.try_cast_into(|s| TCError::unexpected(s, "Library version history"))?;

        // TODO: verify equality of existing versions
        let latest_version = self.subject().latest(*txn.id()).await?;
        for (number, version) in library {
            let number: VersionNumber = number.as_str().parse()?;
            let class = InstanceClass::extend(source.clone(), version);
            if let Some(latest) = latest_version {
                if number > latest {
                    self.put(txn, &[], number.into(), class.into()).await?;
                }
            } else {
                self.put(txn, &[], number.into(), class.into()).await?;
            }
        }

        Ok(())
    }
}

#[async_trait]
impl Replica for BlockChain<crate::cluster::Service> {
    async fn state(&self, txn_id: TxnId) -> TCResult<State> {
        self.subject().state(txn_id).await
    }

    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()> {
        let params = Map::one(label("add"), txn.host().clone().into());

        let state = txn
            .post(source.clone().append(REPLICAS), State::Map(params))
            .await?;

        let library: Map<InstanceClass> =
            state.try_cast_into(|s| TCError::unexpected(s, "Service version history"))?;

        // TODO: verify equality of existing versions
        let latest_version = self.subject().latest(*txn.id()).await?;
        for (number, version) in library {
            let number: VersionNumber = number.as_str().parse()?;
            if let Some(latest) = latest_version {
                if number > latest {
                    self.put(txn, &[], number.into(), version.into()).await?;
                }
            } else {
                self.put(txn, &[], number.into(), version.into()).await?;
            }
        }

        self.subject().replicate(txn, source).await
    }
}

#[async_trait]
impl Replica for BlockChain<CollectionBase> {
    async fn state(&self, _txn_id: TxnId) -> TCResult<State> {
        Ok(State::Chain(self.clone().into()))
    }

    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()> {
        let chain = txn.get(source.append(CHAIN), Value::default()).await?;
        let chain: Self = chain.try_cast_into(|s| {
            bad_request!(
                "blockchain expected to replicate a chain of blocks but found {:?}",
                s,
            )
        })?;

        self.history()
            .replicate(txn, self.subject(), chain.history().clone())
            .await
    }
}

#[async_trait]
impl<T> Replica for SyncChain<T>
where
    T: Persist<CacheBlock, Txn = Txn>
        + Route<State>
        + Restore<CacheBlock>
        + TryCastFrom<State>
        + ToState<State>
        + Transact
        + Clone
        + Send
        + Sync
        + fmt::Debug,
{
    async fn state(&self, _txn_id: TxnId) -> TCResult<State> {
        Ok(self.subject().to_state())
    }

    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()> {
        self.restore_from(txn, source).await
    }
}

#[async_trait]
impl Replica for Chain<CollectionBase> {
    async fn state(&self, txn_id: TxnId) -> TCResult<State> {
        match self {
            Self::Block(chain) => chain.state(txn_id).await,
            Self::Sync(chain) => chain.state(txn_id).await,
        }
    }

    async fn replicate(&self, txn: &Txn, source: Link) -> TCResult<()> {
        debug!("replicate {self:?} from {source}");

        match self {
            Self::Block(chain) => chain.replicate(txn, source).await,
            Self::Sync(chain) => chain.replicate(txn, source).await,
        }
    }
}
