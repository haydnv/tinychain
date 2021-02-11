use std::collections::hash_map::{Entry, HashMap};
use std::sync::Arc;
use std::time::Duration;

use futures::TryFutureExt;
use futures_locks::RwLock;

use error::*;
use generic::{path_label, PathLabel, TCPathBuf};

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
    workspace: fs::Dir,
}

impl TxnServer {
    pub async fn new(workspace: fs::Dir) -> Self {
        let active = RwLock::new(HashMap::new());
        Self { active, workspace }
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
                        gateway.root().into(),
                        txn_id.time().into(),
                        Duration::from_secs(DEFAULT_TTL),
                        actor_id,
                        scopes,
                    );

                    let claims = token.claims();
                    (token, claims)
                };

                let request = Request::new(txn_id, token, claims);
                let txn = Txn::new(gateway, self.workspace.clone(), request);
                entry.insert(txn.clone());
                Ok(txn)
            }
        }
    }

    fn txn_path(txn_id: &TxnId) -> TCPathBuf {
        TCPathBuf::from(PATH).append(txn_id.to_id())
    }
}
