use safecast::TryCastInto;

use tc_btree::BTreeInstance;
use tc_error::TCError;
use tc_transact::Transaction;
use tcgeneric::PathSegment;

use crate::collection::BTree;
use crate::route::{Handler, PutHandler, Route};

struct InsertHandler<'a> {
    btree: &'a BTree,
}

impl<'a> Handler<'a> for InsertHandler<'a> {
    fn put(self: Box<Self>) -> Option<PutHandler<'a>> {
        Some(Box::new(|txn, key, value| {
            Box::pin(async move {
                if key.is_some() {
                    return Err(TCError::bad_request(
                        "BTree::insert does not support an explicit key",
                        key,
                    ));
                }

                let key = value.try_cast_into(|v| TCError::bad_request("invalid BTree key", v))?;
                self.btree.insert(*txn.id(), key).await
            })
        }))
    }
}

impl<'a> From<&'a BTree> for InsertHandler<'a> {
    fn from(btree: &'a BTree) -> Self {
        Self { btree }
    }
}

impl Route for BTree {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.len() != 1 {
            return None;
        }

        match path[0].as_str() {
            "insert" => Some(Box::new(InsertHandler::from(self))),
            _ => None,
        }
    }
}
