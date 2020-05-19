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
    async fn copy_from(_reader: &mut FileCopier, _txn_id: &TxnId, _dest: Arc<Store>) -> Arc<Self> {
        // TODO
        panic!("NOT IMPLEMENTED")
    }

    async fn copy_into(&self, _txn_id: TxnId, _writer: &mut FileCopier) {
        // TODO
        panic!("NOT IMPLEMENTED")
    }

    async fn from_store(_txn_id: &TxnId, _store: Arc<Store>) -> Arc<Self> {
        // TODO
        panic!("NOT IMPLEMENTED")
    }
}
