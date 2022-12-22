use std::convert::TryFrom;

use log::debug;
use safecast::{TryCastFrom, TryCastInto};

use tc_error::*;
use tc_transact::fs::Persist;
use tc_transact::Transaction;
use tc_value::{Link, Value};
use tcgeneric::{Map, NativeClass, PathSegment, TCPath};

use crate::chain::{Chain, ChainType};
use crate::cluster::{service, Service};
use crate::collection::CollectionSchema;
use crate::object::InstanceClass;
use crate::route::{DeleteHandler, GetHandler, Handler, PostHandler, Public, PutHandler, Route};
use crate::scalar::{OpRef, OpRefType, Subject, TCRef};
use crate::state::State;

use super::dir::DirHandler;

impl Route for service::Attr {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        match self {
            Self::Chain(chain) => chain.route(path),
            Self::Scalar(scalar) => scalar.route(path),
        }
    }
}

impl Route for service::Version {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        assert!(!path.is_empty());

        let attr = self.get_attribute(&path[0])?;
        attr.route(&path[1..])
    }
}

struct ServiceHandler<'a> {
    service: &'a Service,
    path: &'a [PathSegment],
}

impl<'a> ServiceHandler<'a> {
    fn new(service: &'a Service, path: &'a [PathSegment]) -> Self {
        Self { service, path }
    }
}

impl<'a> Handler<'a> for ServiceHandler<'a> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if self.path.len() == 0 {
                    return Err(TCError::method_not_allowed(
                        OpRefType::Get,
                        self.service,
                        TCPath::from(self.path),
                    ));
                }

                let number = self.path[0].as_str().parse()?;
                let version = self.service.get_version(*txn.id(), &number).await?;
                version.get(txn, &self.path[1..], key).await
            })
        }))
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key, value| {
            Box::pin(async move {
                if self.path.len() == 0 {
                    return Err(TCError::not_implemented(
                        "create a new version of a Service",
                    ));
                }

                let number = self.path[0].as_str().parse()?;
                let version = self.service.get_version(*txn.id(), &number).await?;
                version.put(txn, &self.path[1..], key, value).await
            })
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, params| {
            Box::pin(async move {
                if self.path.len() == 0 {
                    return Err(TCError::method_not_allowed(
                        OpRefType::Post,
                        self.service,
                        TCPath::from(self.path),
                    ));
                }

                let number = self.path[0].as_str().parse()?;
                let version = self.service.get_version(*txn.id(), &number).await?;
                version.post(txn, &self.path[1..], params).await
            })
        }))
    }

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if self.path.len() == 0 {
                    return Err(TCError::not_implemented("delete a version of a Service"));
                }

                let number = self.path[0].as_str().parse()?;
                let version = self.service.get_version(*txn.id(), &number).await?;
                version.delete(txn, &self.path[1..], key).await
            })
        }))
    }
}

impl Route for Service {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        Some(Box::new(ServiceHandler::new(self, path)))
    }
}

impl<'a> Handler<'a> for DirHandler<'a, Service> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        self.get_entry()
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key, value| {
            Box::pin(async move {
                debug!("{} <- {}: {}", self.dir, key, value);

                let name = key.try_cast_into(|v| {
                    TCError::bad_request("invalid path segment for cluster directory entry", v)
                })?;

                if let Some(_) = self.dir.entry(*txn.id(), &name).await? {
                    return Err(TCError::bad_request(
                        "there is already a directory entry at",
                        name,
                    ))?;
                }

                if Link::can_cast_from(&value) {
                    let link = Link::opt_cast_from(value).expect("service directory host link");
                    return self
                        .create_item_or_dir::<Map<State>>(txn, link, name, None)
                        .await;
                }

                let class = InstanceClass::try_cast_from(value, |s| {
                    TCError::bad_request("invalid Service definition", s)
                })?;

                let (link, mut proto) = class.into_inner();
                let link =
                    link.ok_or_else(|| TCError::bad_request("missing cluster link for", &proto))?;

                let refs = proto
                    .iter()
                    .filter_map(|(name, attr)| if attr.is_ref() { Some(name) } else { None })
                    .cloned()
                    .collect::<Vec<_>>();

                let mut version = Map::<State>::new();
                let mut classes = Map::<InstanceClass>::new();

                for name in refs {
                    let attr = proto.remove(&name).expect("service attr");
                    let tc_ref = TCRef::try_from(attr)?;
                    match tc_ref {
                        TCRef::Op(OpRef::Post((Subject::Link(link), proto))) => {
                            classes.insert(name, InstanceClass::anonymous(Some(link), proto));
                        }
                        TCRef::Op(OpRef::Get((chain_type, collection))) => {
                            let chain_type = resolve_type::<ChainType>(chain_type)?;
                            let schema = TCRef::try_from(collection)?;
                            let schema = CollectionSchema::from_scalar(schema)?;
                            let store = txn.context().create_store(*txn.id(), name.clone()).await?;

                            let chain = Chain::create(txn, (chain_type, schema), store).await?;
                            version.insert(name, State::Chain(chain));
                        }
                        other => {
                            return Err(TCError::bad_request("invalid Service attribute", other));
                        }
                    }
                }

                if !classes.is_empty() {
                    return Err(TCError::not_implemented(format!(
                        "install Class dependencies {}",
                        classes
                    )));
                }

                version.extend(
                    proto
                        .into_iter()
                        .map(|(name, op)| (name, State::Scalar(op))),
                );

                self.create_item_or_dir(txn, link, name, Some(version))
                    .await
            })
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(self.method_not_allowed::<Map<State>, State>(OpRefType::Post))
    }

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(self.method_not_allowed::<Value, ()>(OpRefType::Delete))
    }
}

fn resolve_type<T: NativeClass>(subject: Subject) -> TCResult<T> {
    match subject {
        Subject::Link(link) if link.host().is_none() => {
            T::from_path(link.path()).ok_or_else(|| {
                TCError::unsupported(format!(
                    "{} is not a {}",
                    link.path(),
                    std::any::type_name::<T>()
                ))
            })
        }
        Subject::Link(link) => Err(TCError::not_implemented(format!(
            "support for a user-defined Class of {} in a Service: {}",
            std::any::type_name::<T>(),
            link
        ))),
        subject => Err(TCError::bad_request(
            format!("expected a {} but found", std::any::type_name::<T>()),
            subject,
        )),
    }
}
