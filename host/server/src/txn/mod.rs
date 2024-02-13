use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use freqfs::DirLock;
use futures::Future;
use safecast::CastInto;
use umask::Mode;
use uuid::Uuid;

use tc_error::*;
use tc_transact::public::StateInstance;
use tc_transact::{RPCClient, Transaction, TxnId};
use tc_value::{Link, ToUrl, Value};
use tcgeneric::{Id, PathSegment, ThreadSafe};

use crate::SignedToken;

pub use hypothetical::Hypothetical;
pub use server::TxnServer;

mod hypothetical;
mod server;

#[derive(Clone)]
pub(super) enum LazyDir<FE> {
    Workspace(DirLock<FE>),
    Lazy(Arc<Self>, Id),
}

impl<FE: Send + Sync> LazyDir<FE> {
    fn get_or_create<'a>(
        &'a self,
        txn_id: &'a TxnId,
    ) -> Pin<Box<dyn Future<Output = TCResult<DirLock<FE>>> + Send + 'a>> {
        Box::pin(async move {
            match self {
                Self::Workspace(workspace) => {
                    let mut parent = workspace.write().await;

                    parent
                        .get_or_create_dir(txn_id.to_string())
                        .map_err(TCError::from)
                }
                Self::Lazy(parent, name) => {
                    let parent = parent.get_or_create(txn_id).await?;
                    let mut parent = parent.write().await;

                    parent
                        .get_or_create_dir(name.to_string())
                        .map_err(TCError::from)
                }
            }
        })
    }

    fn create_dir(self, name: Id) -> Self {
        Self::Lazy(Arc::new(self), name)
    }

    fn create_dir_unique(self) -> Self {
        Self::Lazy(Arc::new(self), Uuid::new_v4().into())
    }
}

impl<FE> From<DirLock<FE>> for LazyDir<FE> {
    fn from(dir: DirLock<FE>) -> Self {
        Self::Workspace(dir)
    }
}

#[derive(Clone)]
pub struct Txn<FE> {
    id: TxnId,
    workspace: LazyDir<FE>,
    token: Option<SignedToken>,
}

impl<FE> Txn<FE> {
    pub(super) fn new(workspace: LazyDir<FE>, id: TxnId, token: Option<SignedToken>) -> Self {
        Self {
            id,
            workspace,
            token,
        }
    }

    pub fn mode(&self, actor: &Link, path: &[PathSegment]) -> Mode {
        todo!("Txn::mode")
    }

    pub fn is_leader(&self, path: &[PathSegment]) -> bool {
        todo!("Txn::is_leader")
    }
}

#[async_trait]
impl<FE: ThreadSafe + Clone> Transaction<FE> for Txn<FE> {
    #[inline]
    fn id(&self) -> &TxnId {
        &self.id
    }

    async fn context(&self) -> TCResult<DirLock<FE>> {
        self.workspace.get_or_create(&self.id).await
    }

    fn subcontext<I: Into<Id> + Send>(&self, id: I) -> Self {
        Txn {
            workspace: self.workspace.clone().create_dir(id.into()),
            id: self.id,
            token: self.token.clone(),
        }
    }

    fn subcontext_unique(&self) -> Self {
        Txn {
            workspace: self.workspace.clone().create_dir_unique(),
            id: self.id,
            token: self.token.clone(),
        }
    }
}

#[async_trait]
impl<FE, State> RPCClient<State> for Txn<FE>
where
    FE: ThreadSafe + Clone,
    State: StateInstance<FE = FE, Txn = Self>,
{
    async fn get<'a, L, V>(&'a self, link: L, key: V) -> TCResult<State>
    where
        L: Into<ToUrl<'a>> + Send,
        V: CastInto<Value> + Send,
    {
        todo!()
    }

    async fn put<'a, L, K, V>(&'a self, link: L, key: K, value: V) -> TCResult<()>
    where
        L: Into<ToUrl<'a>> + Send,
        K: CastInto<Value> + Send,
        V: CastInto<State> + Send,
    {
        todo!()
    }

    async fn post<'a, L, P>(&'a self, link: L, params: P) -> TCResult<State>
    where
        L: Into<ToUrl<'a>> + Send,
        P: CastInto<State> + Send,
    {
        todo!()
    }

    async fn delete<'a, L, V>(&'a self, link: L, key: V) -> TCResult<()>
    where
        L: Into<ToUrl<'a>> + Send,
        V: CastInto<Value> + Send,
    {
        todo!()
    }
}
