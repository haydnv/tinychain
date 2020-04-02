use std::sync::Arc;
use std::time;

use crate::context::*;
use crate::dir::{Dir, DirEntry};
use crate::drive::Drive;
use crate::error;
use crate::state::block::BlockContext;
use crate::state::chain::ChainContext;
use crate::state::table::TableContext;
use crate::state::value::ValueContext;
use crate::transaction::Transaction;

pub struct Host {
    root: Arc<Dir>,
}

impl Host {
    pub fn new(workspace: Arc<Drive>) -> TCResult<Arc<Host>> {
        let kernel = Dir::new();
        kernel
            .clone()
            .put_exe(Link::to("/block")?, BlockContext::new(workspace));
        kernel
            .clone()
            .put_exe(Link::to("/chain")?, ChainContext::new());
        kernel
            .clone()
            .put_exe(Link::to("/table")?, TableContext::new());
        kernel
            .clone()
            .put_dir(Link::to("/value")?, ValueContext::init()?);

        let root = Dir::new();
        root.clone().put_dir(Link::to("/sbin")?, kernel);

        Ok(Arc::new(Host { root }))
    }

    pub fn time(&self) -> u128 {
        time::SystemTime::now()
            .duration_since(time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    }

    pub fn new_transaction(self: Arc<Self>) -> TCResult<Arc<Transaction>> {
        Transaction::new(self)
    }

    fn route(
        self: Arc<Self>,
        txn: Arc<Transaction>,
        path: Link,
    ) -> TCResult<(DirEntry, Link, Link)> {
        let mut pwd = self.root.clone();
        for i in 0..path.len() {
            let (from, to) = path.split(i)?;
            let dir_entry = pwd.clone().get(txn.clone(), &path.segment(i))?;
            if let DirEntry::Dir(d) = dir_entry {
                pwd = d.clone();
            } else {
                return Ok((dir_entry, from, to));
            }
        }

        Err(error::not_found(path))
    }

    pub async fn get(self: Arc<Self>, txn: Arc<Transaction>, path: Link) -> TCResult<Arc<TCState>> {
        match self.route(txn.clone(), path)? {
            (dir_entry, from, to) => match dir_entry {
                DirEntry::Dir(_) => Err(error::method_not_allowed(from)),
                DirEntry::Executable(_) => Err(error::method_not_allowed(from)),
                DirEntry::Context(cxt) => cxt.get(txn, to).await,
                DirEntry::Object(obj) => obj.get(txn, to).await,
            },
        }
    }

    pub async fn put(
        self: Arc<Self>,
        txn: Arc<Transaction>,
        path: Link,
        value: TCValue,
    ) -> TCResult<()> {
        let (dir_entry, from, to) = self.route(txn.clone(), path)?;
        if to != Link::new() {
            return Err(error::method_not_allowed(from));
        }

        match dir_entry {
            DirEntry::Dir(_) => Err(error::method_not_allowed(from)),
            DirEntry::Executable(_) => Err(error::method_not_allowed(from)),
            DirEntry::Context(cxt) => cxt.put(txn, value).await,
            DirEntry::Object(obj) => obj.put(txn, value).await,
        }
    }

    pub async fn post(
        self: Arc<Self>,
        txn: Arc<Transaction>,
        path: Link,
    ) -> TCResult<Arc<TCState>> {
        match self.route(txn.clone(), path)? {
            (dir_entry, from, to) => match dir_entry {
                DirEntry::Dir(_) => Err(error::method_not_allowed(from)),
                DirEntry::Context(_) => Err(error::method_not_allowed(from)),
                DirEntry::Object(obj) => obj.post(txn, to).await,
                DirEntry::Executable(exe) => exe.post(txn, to).await,
            },
        }
    }
}
