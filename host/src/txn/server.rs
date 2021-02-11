use std::collections::hash_map::{Entry, HashMap};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use futures::TryFutureExt;
use futures_locks::RwLock;
use tokio::sync::mpsc;

use error::*;
use generic::{path_label, PathLabel, TCPathBuf};
use transact::Transact;

use crate::fs;
use crate::gateway::Gateway;
use crate::scalar::Link;

use super::request::*;
use super::{Txn, TxnId};

const DEFAULT_TTL: u64 = 30;
const PATH: PathLabel = path_label(&["host", "txn"]);

#[derive(Clone)]
pub struct TxnServer {
    active: RwLock<HashMap<TxnId, Txn>>,
    sender: mpsc::UnboundedSender<TxnId>,
    workspace: fs::Dir,
}

impl TxnServer {
    pub async fn new(workspace: fs::Dir) -> Self {
        let (sender, mut receiver) = mpsc::unbounded_channel();

        let active = RwLock::new(HashMap::new());
        let active_clone = active.clone();
        let workspace_clone = workspace.clone();
        thread::spawn(move || {
            use futures::executor::block_on;

            while let Some(txn_id) = block_on(receiver.recv()) {
                let txn: Option<Txn> = { block_on(active_clone.write()).remove(&txn_id) };
                if let Some(txn) = txn {
                    // TODO: implement delete
                    // block_on(workspace_clone.delete(txn_id, txn_id.to_path())).unwrap();
                    block_on(txn.finalize(&txn_id));
                    block_on(workspace_clone.finalize(&txn_id));
                }
            }
        });

        Self {
            active,
            sender,
            workspace,
        }
    }

    pub async fn new_txn(
        &self,
        gateway: Arc<Gateway>,
        txn_id: TxnId,
        token: Option<String>,
    ) -> TCResult<Txn> {
        let mut active = self.active.write().await;

        let actor_id = Link::from(Self::txn_path(&txn_id)).into();

        match active.entry(txn_id) {
            Entry::Occupied(entry) => {
                let txn = entry.get();
                // TODO: authorize access to this Txn
                Ok(txn.clone())
            }
            Entry::Vacant(entry) => {
                use rjwt::Resolve;

                let scopes: Vec<Scope> = vec![SCOPE_ROOT.into()];
                let (token, claims) = if let Some(token) = token {
                    Resolver::new(&gateway, &txn_id)
                        .consume(actor_id, scopes, token, txn_id.time().into())
                        .map_err(TCError::unauthorized)
                        .await?
                } else {
                    let token: Token = <Token as TokenExt>::new(
                        gateway.root().clone().into(),
                        txn_id.time().into(),
                        Duration::from_secs(DEFAULT_TTL),
                        actor_id,
                        scopes,
                    );

                    let claims = token.claims();
                    (token, claims)
                };

                let request = Request::new(txn_id, token, claims);
                let txn = Txn::new(
                    self.sender.clone(),
                    gateway,
                    self.workspace.clone(),
                    request,
                );
                entry.insert(txn.clone());
                Ok(txn)
            }
        }
    }

    fn txn_path(txn_id: &TxnId) -> TCPathBuf {
        TCPathBuf::from(PATH).append(txn_id.to_id())
    }
}
