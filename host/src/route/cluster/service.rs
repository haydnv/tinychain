use log::{debug, trace};
use safecast::{TryCastFrom, TryCastInto};

use tc_error::*;
use tc_transact::Transaction;
use tc_value::{Value, Version as VersionNumber};
use tcgeneric::{Id, Map, PathSegment, TCPath, TCPathBuf, Tuple};

use crate::cluster::{service, DirItem, Replica, Service};
use crate::object::InstanceClass;
use crate::route::{DeleteHandler, GetHandler, Handler, PostHandler, Public, PutHandler, Route};
use crate::scalar::{OpRef, OpRefType, Scalar, Subject, TCRef};
use crate::state::State;
use crate::txn::Txn;
use crate::CLASS;

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
        debug!("service::Version::route {}", TCPath::from(path));

        assert!(!path.is_empty());

        match self.get_attribute(&path[0]) {
            Some(attr) => {
                trace!("found attr {} at {}", attr, &path[0]);
                attr.route(&path[1..])
            }
            None => {
                trace!(
                    "{} has no attr {} (attrs are {})",
                    self,
                    &path[0],
                    self.attrs().collect::<Tuple<&Id>>()
                );

                None
            }
        }
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

    fn create_version<'b>(self: Box<Self>) -> PutHandler<'a, 'b>
    where
        'b: 'a,
    {
        Box::new(|txn, key, value| {
            Box::pin(async move {
                debug!("{} <- {}: {}", self.service, key, value);

                let number = VersionNumber::try_cast_from(key, |v| {
                    TCError::bad_request("invalid version number", v)
                })?;

                let class = InstanceClass::try_cast_from(value, |v| {
                    TCError::bad_request("invalid Class", v)
                })?;

                let (link, version) = class.into_inner();
                let link =
                    link.ok_or_else(|| TCError::bad_request("missing cluster link for", &version))?;

                let mut classes = Map::new();
                let mut schema = Map::new();

                for (name, scalar) in version {
                    match scalar {
                        Scalar::Ref(tc_ref) => match *tc_ref {
                            TCRef::Op(OpRef::Post((Subject::Link(classpath), proto)))
                                if !proto.is_empty() =>
                            {
                                let class = InstanceClass::anonymous(Some(classpath), proto);
                                classes.insert(name, class);
                            }
                            tc_ref => {
                                schema.insert(name, Scalar::from(tc_ref));
                            }
                        },
                        scalar => {
                            schema.insert(name, scalar);
                        }
                    }
                }

                if !classes.is_empty() && txn.is_leader(link.path()) {
                    let mut class_path = TCPathBuf::from(CLASS);
                    class_path.extend(link.path()[1..].iter().cloned());

                    txn.put(class_path.into(), number.clone().into(), classes.into())
                        .await?;
                }

                let schema = InstanceClass::anonymous(Some(link.clone()), schema);
                let version = self
                    .service
                    .create_version(txn, number.clone(), schema)
                    .await?;

                if link != txn.link(link.path().clone()) {
                    version.replicate(txn, link.append(number)).await?;
                }

                Ok(())
            })
        })
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
        if self.path.is_empty() {
            Some(self.create_version())
        } else {
            Some(Box::new(|txn: &Txn, key: Value, value: State| {
                Box::pin(async move {
                    let number = self.path[0].as_str().parse()?;
                    let version = self.service.get_version(*txn.id(), &number).await?;
                    version.put(txn, &self.path[1..], key, value).await
                })
            }))
        }
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
        debug!("Service::route {}", TCPath::from(path));
        Some(Box::new(ServiceHandler::new(self, path)))
    }
}

impl<'a> Handler<'a> for DirHandler<'a, Service> {
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

                let class = InstanceClass::try_cast_from(value, |v| {
                    TCError::bad_request("invalid Class", v)
                })?;

                let (link, version) = class.into_inner();
                let link =
                    link.ok_or_else(|| TCError::bad_request("missing cluster link for", &version))?;

                if version.is_empty() {
                    self.create_item_or_dir::<Map<State>>(txn, link, name, None)
                        .await
                } else {
                    let version = InstanceClass::anonymous(Some(link.clone()), version);

                    self.create_item_or_dir(txn, link, name, Some(version))
                        .await
                }
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
