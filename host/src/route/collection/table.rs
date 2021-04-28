use std::convert::TryFrom;

use futures::TryFutureExt;
use log::debug;
use safecast::*;

use tc_btree::Node;
use tc_error::*;
use tc_table::{Bounds, ColumnBound, TableInstance};
use tc_transact::Transaction;
use tc_value::{Range, Value};
use tcgeneric::{Map, PathSegment};

use crate::collection::{Collection, Table};
use crate::fs::{Dir, File};
use crate::route::{GetHandler, Handler, PostHandler, PutHandler, Route};
use crate::scalar::Scalar;
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

struct TableHandler<'a, T> {
    table: &'a T,
}

impl<'a, T: TableInstance<File<Node>, Dir, Txn>> Handler<'a> for TableHandler<'a, T> {
    fn put(self: Box<Self>) -> Option<PutHandler<'a>> {
        Some(Box::new(|txn, key, values| {
            Box::pin(async move {
                let key =
                    key.try_cast_into(|k| TCError::bad_request("invalid key for Table row", k))?;

                let values = values
                    .try_cast_into(|v| TCError::bad_request("invalid values for Table row", v))?;

                self.table.insert(*txn.id(), key, values).await
            })
        }))
    }

    fn post(self: Box<Self>) -> Option<PostHandler<'a>> {
        Some(Box::new(|_txn, params| {
            Box::pin(async move {
                let bounds = Scalar::try_cast_from(State::Map(params), |s| {
                    TCError::bad_request("invalid Table bounds", s)
                })?;

                let bounds = cast_into_bounds(bounds)?;
                debug!("slice Table with bounds {}", bounds);

                let table = self.table.clone().slice(bounds).map(|slice| slice.into())?;
                Ok(Collection::Table(table).into())
            })
        }))
    }
}

impl<'a, T> From<&'a T> for TableHandler<'a, T> {
    fn from(table: &'a T) -> Self {
        Self { table }
    }
}

impl Route for Table {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        if path.is_empty() {
            Some(Box::new(TableHandler::from(self)))
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

#[inline]
fn cast_into_bounds(scalar: Scalar) -> TCResult<Bounds> {
    let scalar = Map::<Scalar>::try_from(scalar)?;
    scalar
        .into_iter()
        .map(|(col_name, bound)| {
            if bound.matches::<Range>() {
                Ok(ColumnBound::In(bound.opt_cast_into().unwrap()))
            } else if bound.matches::<Value>() {
                Ok(ColumnBound::Is(bound.opt_cast_into().unwrap()))
            } else {
                Err(TCError::bad_request("invalid column bound", bound))
            }
            .map(|bound| (col_name, bound))
        })
        .collect()
}
