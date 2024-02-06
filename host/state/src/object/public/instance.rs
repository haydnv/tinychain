use std::fmt;

use futures::future;
use log::{debug, info};
use safecast::AsType;
use tc_chain::ChainBlock;
use tc_collection::{BTreeNode, DenseCacheFile, TensorNode};

use tc_error::*;
use tc_transact::public::generic::COPY;
use tc_transact::public::{GetHandler, Handler, Route, ToState};
use tc_transact::{fs, RPCClient, Transaction};
use tcgeneric::{Instance, PathSegment, TCPath};

use crate::object::InstanceExt;
use crate::State;

use super::method::route_attr;

struct CopyHandler<'a, T> {
    instance: &'a T,
}

impl<'a, Txn, FE, T> Handler<'a, State<Txn, FE>> for CopyHandler<'a, T>
where
    Txn: Transaction<FE> + RPCClient<State<Txn, FE>>,
    FE: DenseCacheFile
        + AsType<BTreeNode>
        + AsType<ChainBlock>
        + AsType<TensorNode>
        + for<'b> fs::FileSave<'b>
        + Clone,
    T: Instance + fmt::Debug + 'a,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State<Txn, FE>>>
    where
        'b: 'a,
    {
        Some(Box::new(move |_txn, _key| {
            Box::pin(future::ready(Err(not_implemented!(
                "{:?} has no /copy method",
                self.instance
            ))))
        }))
    }
}

impl<'a, T> From<&'a T> for CopyHandler<'a, T> {
    fn from(instance: &'a T) -> Self {
        Self { instance }
    }
}

impl<Txn, FE, T> Route<State<Txn, FE>> for InstanceExt<Txn, FE, T>
where
    Txn: Transaction<FE> + RPCClient<State<Txn, FE>>,
    FE: DenseCacheFile
        + AsType<BTreeNode>
        + AsType<ChainBlock>
        + AsType<TensorNode>
        + for<'a> fs::FileSave<'a>
        + Clone,
    T: ToState<State<Txn, FE>> + Instance + Route<State<Txn, FE>> + fmt::Debug,
    Self: ToState<State<Txn, FE>>,
{
    fn route<'a>(
        &'a self,
        path: &'a [PathSegment],
    ) -> Option<Box<dyn Handler<'a, State<Txn, FE>> + 'a>> {
        debug!(
            "{:?} with members {:?} route {} (parent is {} {:?})",
            self,
            self.members(),
            TCPath::from(path),
            std::any::type_name::<T>(),
            self.parent()
        );

        if path.is_empty() {
            debug!("routing to parent: {:?}", self.parent());

            if let Some(handler) = self.parent().route(path) {
                Some(handler)
            } else if path == &COPY[..] {
                info!("tried to copy an public with no /copy method implemented");
                Some(Box::new(CopyHandler::from(self)))
            } else {
                debug!("{:?} has no handler for {}", self, TCPath::from(path));
                None
            }
        } else if let Some(attr) = self.members().get(&path[0]) {
            debug!("{} found in {:?} members: {:?}", &path[0], self, attr);

            if let State::Scalar(attr) = attr {
                route_attr(self, &path[0], attr, &path[1..])
            } else {
                attr.route(&path[1..])
            }
        } else if let Some(attr) = self.proto().get(&path[0]) {
            debug!("{} found in instance proto", &path[0]);
            route_attr(self, &path[0], attr, &path[1..])
        } else if let Some(handler) = self.parent().route(path) {
            debug!("{} found in parent", TCPath::from(path));
            Some(handler)
        } else if let Some(attr) = self.proto().get(&path[0]) {
            debug!("{} found in class proto", path[0]);
            attr.route(&path[1..])
        } else {
            debug!(
                "not found in {:?}: {} (while resolving {})",
                self,
                &path[0],
                TCPath::from(path)
            );

            None
        }
    }
}
