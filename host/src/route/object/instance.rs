use std::fmt;

use futures::future;
use log::{debug, info};

use tc_error::*;
use tc_transact::public::generic::COPY;
use tc_transact::public::{Handler, Route, ToState};
use tcgeneric::{Instance, PathSegment, TCPath};

use crate::object::InstanceExt;
use crate::state::State;

use super::method::route_attr;
use super::GetHandler;

struct CopyHandler<'a, T> {
    instance: &'a T,
}

impl<'a, T> Handler<'a, State> for CopyHandler<'a, T>
where
    T: Instance + fmt::Debug + 'a,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
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

impl<T: ToState<State> + Instance + Route<State> + fmt::Debug> Route<State> for InstanceExt<T>
where
    Self: ToState<State>,
{
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
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
                info!("tried to copy an object with no /copy method implemented");
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
