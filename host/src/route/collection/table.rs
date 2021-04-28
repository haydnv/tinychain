use futures::TryFutureExt;

use tc_btree::Node;
use tc_error::*;
use tc_table::TableInstance;
use tc_transact::Transaction;
use tcgeneric::PathSegment;

use crate::collection::Table;
use crate::fs::{Dir, File};
use crate::route::{GetHandler, Handler, Route};
use crate::state::State;
use crate::txn::Txn;

struct CountHandler<T> {
    table: T,
}

impl<'a, T: TableInstance<File<Node>, Dir, Txn> + 'a> Handler<'a> for CountHandler<T> {
    fn get(self: Box<Self>) -> Option<GetHandler<'a>> {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if key.is_some() {
                    return Err(TCError::bad_request(
                        "Table::count does not accept a key (call Table::slice first)",
                        key,
                    ));
                }

                self.table.count(*txn.id()).map_ok(State::from).await
            })
        }))
    }
}

impl<T> From<T> for CountHandler<T> {
    fn from(table: T) -> Self {
        Self { table }
    }
}

impl Route for Table {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.is_empty() {
            // TODO
            return None;
        } else if path.len() == 1 {
            match path[0].as_str() {
                "count" => Some(Box::new(CountHandler::from(self.clone()))),
                _ => None,
            }
        } else {
            None
        }
    }
}
