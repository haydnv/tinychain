use std::sync::Arc;

use async_trait::async_trait;

use crate::error;
use crate::internal::block::Store;
use crate::internal::file::*;
use crate::object::actor::{Actor, Token};
use crate::object::TCObject;
use crate::state::*;
use crate::transaction::{Txn, TxnId};
use crate::value::link::TCPath;
use crate::value::{TCResult, TCValue};

pub struct Cluster {
    actors: Arc<table::Table>,
    hosts: Arc<table::Table>,
    hosted: Arc<Directory>,
}

#[async_trait]
impl Collection for Cluster {
    type Key = TCPath;
    type Value = State;

    async fn get(
        self: &Arc<Self>,
        _txn: &Arc<Txn<'_>>,
        path: &TCPath,
        _auth: &Option<Token>,
    ) -> TCResult<Self::Value> {
        if path.len() != 1 {
            return Err(error::not_found(path));
        }

        match path[0].as_str() {
            "actors" => Ok(self.actors.clone().into()),
            "hosts" => Ok(self.hosts.clone().into()),
            "hosted" => Ok(self.hosted.clone().into()),
            _ => Err(error::not_found(path)),
        }
    }

    async fn put(
        self: Arc<Self>,
        txn: &Arc<Txn<'_>>,
        path: TCPath,
        state: State,
        auth: &Option<Token>,
    ) -> TCResult<Arc<Self>> {
        if path[0] == "hosted" {
            if path.len() > 1 {
                self.hosted
                    .clone()
                    .put(txn, path.slice_from(1), state, auth)
                    .await?;
                Ok(self)
            } else {
                Err(error::forbidden(
                    "You are not allowed to perform this action",
                ))
            }
        } else {
            Err(error::method_not_allowed(path))
        }
    }
}

#[async_trait]
impl Persistent for Cluster {
    type Config = TCValue;

    async fn create(txn: &Arc<Txn<'_>>, _: TCValue) -> TCResult<Arc<Cluster>> {
        let actors =
            table::Table::create(&txn.subcontext("actors".parse()?).await?, Actor::schema())
                .await?;

        let hosts = table::Table::create(
            &txn.subcontext("hosts".parse()?).await?,
            table::Schema::from(
                vec![("address".parse()?, "/sbin/value/link/address".parse()?)],
                vec![],
                "1.0.0".parse().unwrap(),
            ),
        )
        .await?;

        let hosted = Directory::new(
            &txn.id(),
            txn.context()
                .reserve(&txn.id(), "hosted".parse().unwrap())
                .await?,
        )
        .await?;

        Ok(Arc::new(Cluster {
            actors,
            hosts,
            hosted,
        }))
    }
}

#[async_trait]
impl File for Cluster {
    async fn copy_from(reader: &mut FileCopier, txn_id: &TxnId, dest: Arc<Store>) -> Arc<Self> {
        println!("Cluster::copy_from actors");
        let actors = table::Table::copy_from(
            reader,
            txn_id,
            dest.reserve(txn_id, "actors".parse().unwrap())
                .await
                .unwrap(),
        )
        .await;

        println!("Cluster::copy_from hosts");
        let hosts = table::Table::copy_from(
            reader,
            txn_id,
            dest.reserve(txn_id, "hosts".parse().unwrap())
                .await
                .unwrap(),
        )
        .await;

        println!("Cluster::copy_from hosted");
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
        println!("Cluster::copy_into actors");
        self.actors.copy_into(txn_id.clone(), writer).await;

        println!("Cluster::copy_into hosts");
        self.hosts.copy_into(txn_id.clone(), writer).await;

        println!("Cluster::copy_into hosted");
        self.hosted.copy_into(txn_id.clone(), writer).await;
    }

    async fn from_store(txn_id: &TxnId, store: Arc<Store>) -> Arc<Self> {
        let actors = table::Table::from_store(
            txn_id,
            store
                .get_store(txn_id, &"actors".parse().unwrap())
                .await
                .unwrap(),
        )
        .await;

        let hosts = table::Table::from_store(
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
