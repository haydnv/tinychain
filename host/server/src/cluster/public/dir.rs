use std::fmt;

use safecast::TryCastInto;

use tc_error::*;
use tc_state::CacheBlock;
use tc_transact::fs;
use tc_transact::public::{Handler, PutHandler, Route, StateInstance};
use tcgeneric::{PathSegment, TCPath};

use crate::cluster::dir::{Dir, DirCreate, DirEntry};
use crate::cluster::{Cluster, Schema};
use crate::{State, Txn};

struct DirHandler<'a, T> {
    dir: &'a Dir<T>,
}

impl<'a, T> Handler<'a, State> for DirHandler<'a, T>
where
    T: Send + Sync + fmt::Debug,
    Dir<T>: fs::Persist<CacheBlock, Txn = Txn, Schema = Schema> + Route<State> + fmt::Debug,
    DirEntry<T>: Clone,
    Cluster<T>: fs::Persist<CacheBlock, Txn = Txn, Schema = Schema>,
    Cluster<Dir<T>>: fs::Persist<CacheBlock, Txn = Txn, Schema = Schema> + Clone,
{
    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key, value| {
            Box::pin(async move {
                let name: PathSegment =
                    key.try_cast_into(|v| TCError::unexpected(v, "a directory name"))?;

                if value.is_map() {
                    if value.is_none() {
                        self.dir.create_dir(txn, name).await?;
                        Ok(())
                    } else {
                        Err(TCError::unexpected(value, "an empty Map"))
                    }
                } else {
                    Err(not_implemented!("create a new item in a cluster directory"))
                }
            })
        }))
    }
}

impl<'a, T> From<&'a Dir<T>> for DirHandler<'a, T> {
    fn from(dir: &'a Dir<T>) -> Self {
        Self { dir }
    }
}

impl<T> Route<State> for Dir<T>
where
    T: Send + Sync + fmt::Debug,
    Dir<T>: fs::Persist<CacheBlock, Txn = Txn, Schema = Schema> + fmt::Debug,
    DirEntry<T>: Clone,
    Cluster<T>: fs::Persist<CacheBlock, Txn = Txn, Schema = Schema>,
    Cluster<Dir<T>>: fs::Persist<CacheBlock, Txn = Txn, Schema = Schema> + Clone,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        if path.is_empty() {
            Some(Box::new(DirHandler::from(self)))
        } else {
            unreachable!("request to {} routed to parent dir", TCPath::from(path))
        }
    }
}
