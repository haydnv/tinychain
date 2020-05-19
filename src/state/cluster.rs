use std::sync::Arc;

use async_trait::async_trait;

use crate::error;
use crate::internal::block::Store;
use crate::internal::file::*;
use crate::internal::Directory;
use crate::object::actor::Token;
use crate::state::{Collection, File, Persistent, Schema, State, Table};
use crate::transaction::{Txn, TxnId};
use crate::value::link::TCPath;
use crate::value::{TCResult, TCValue};

pub struct Cluster {
    actors: Arc<Table>,
    hosts: Arc<Table>,
    hosted: Arc<Directory>,
}

#[async_trait]
impl Collection for Cluster {
    type Key = TCPath;
    type Value = State;

    async fn get(
        self: &Arc<Self>,
        _txn: &Arc<Txn<'_>>,
        _path: &TCPath,
        _auth: &Option<Token>,
    ) -> TCResult<Self::Value> {
        Err(error::not_implemented())
    }

    async fn put(
        self: Arc<Self>,
        _txn: &Arc<Txn<'_>>,
        _path: TCPath,
        _state: State,
        _auth: &Option<Token>,
    ) -> TCResult<Arc<Self>> {
        Err(error::not_implemented())
    }
}

#[async_trait]
impl Persistent for Cluster {
    type Config = TCValue;

    async fn create(txn: &Arc<Txn<'_>>, _: TCValue) -> TCResult<Arc<Cluster>> {
        let actors = Table::create(
            &txn.subcontext("actors".parse()?).await?,
            Schema::from(
                vec![("actor".parse()?, "/sbin/object/actor".parse()?)],
                vec![],
                "1.0.0".parse()?,
            ),
        )
        .await?;

        let hosts = Table::create(
            &txn.subcontext("hosts".parse()?).await?,
            Schema::from(
                vec![("address".parse()?, "/sbin/value/link/address".parse()?)],
                vec![],
                "1.0.0".parse().unwrap(),
            ),
        )
        .await?;

        Ok(Arc::new(Cluster {
            actors,
            hosts,
            hosted: Directory::new(
                &txn.id(),
                txn.context()
                    .reserve(&txn.id(), "hosted".parse().unwrap())
                    .await?,
            )
            .await?,
        }))
    }
}

#[async_trait]
impl File for Cluster {
    async fn copy_from(reader: &mut FileCopier, txn_id: &TxnId, dest: Arc<Store>) -> Arc<Self> {
        let actors = Table::copy_from(
            reader,
            txn_id,
            dest.reserve(txn_id, "actors".parse().unwrap())
                .await
                .unwrap(),
        )
        .await;

        let hosts = Table::copy_from(
            reader,
            txn_id,
            dest.reserve(txn_id, "hosts".parse().unwrap())
                .await
                .unwrap(),
        )
        .await;

        let hosted = Directory::copy_from(
            reader,
            txn_id,
            dest.reserve(txn_id, "hosted".parse().unwrap())
                .await
                .unwrap(),
        )
        .await;

        Arc::new(Cluster {
            actors,
            hosts,
            hosted,
        })
    }

    async fn copy_into(&self, txn_id: TxnId, writer: &mut FileCopier) {
        self.actors.copy_into(txn_id.clone(), writer).await;
        self.hosts.copy_into(txn_id.clone(), writer).await;
        self.hosted.copy_into(txn_id.clone(), writer).await;
    }

    async fn from_store(txn_id: &TxnId, store: Arc<Store>) -> Arc<Self> {
        let actors = Table::from_store(
            txn_id,
            store
                .get_store(txn_id, &"actors".parse().unwrap())
                .await
                .unwrap(),
        )
        .await;

        let hosts = Table::from_store(
            txn_id,
            store
                .get_store(txn_id, &"hosts".parse().unwrap())
                .await
                .unwrap(),
        )
        .await;

        let hosted = Directory::from_store(
            txn_id,
            store
                .get_store(txn_id, &"hosted".parse().unwrap())
                .await
                .unwrap(),
        )
        .await;

        Arc::new(Cluster {
            actors,
            hosts,
            hosted,
        })
    }
}
