use log::{debug, trace};
use safecast::{TryCastFrom, TryCastInto};

use tc_error::*;
use tc_scalar::{OpRef, Scalar, Subject, TCRef};
use tc_state::object::public::method::route_attr;
use tc_state::object::InstanceClass;
use tc_transact::public::helpers::MethodNotAllowedHandler;
use tc_transact::public::{
    DeleteHandler, GetHandler, Handler, PostHandler, Public, PutHandler, Route,
};
use tc_transact::{Replicate, Transaction};
use tc_value::{Value, Version as VersionNumber};
use tcgeneric::{Id, Map, PathSegment, TCPath};

use crate::cluster::dir::DirItem;
use crate::cluster::service::{Attr, Service, Version};
use crate::{State, Txn};

struct ServiceHandler<'a> {
    service: &'a Service,
}

impl<'a> Handler<'a, State> for ServiceHandler<'a> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if key.is_some() {
                    let number: Id =
                        key.try_cast_into(|v| TCError::unexpected(v, "a semantic version number"))?;

                    let version = self.service.get_version_proto(*txn.id(), &number).await?;

                    Ok(version.into())
                } else {
                    let version_numbers = self.service.list_versions(*txn.id()).await?;
                    let version_numbers = version_numbers.into_iter().map(Id::from).collect();
                    Ok(Value::Tuple(version_numbers).into())
                }
            })
        }))
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key, value| {
            Box::pin(async move {
                debug!("create new Service version {}", key);

                let number = VersionNumber::try_cast_from(key, |v| {
                    TCError::unexpected(v, "a version number")
                })?;

                let class =
                    InstanceClass::try_cast_from(value, |v| TCError::unexpected(v, "a Class"))?;

                let (link, version) = class.into_inner();

                let mut schema = Map::new();

                for (name, scalar) in version {
                    match scalar {
                        Scalar::Ref(tc_ref) => match *tc_ref {
                            TCRef::Op(OpRef::Post((Subject::Link(_classpath), proto)))
                                if !proto.is_empty() =>
                            {
                                // it's a class
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

                let schema = InstanceClass::extend(link.clone(), schema);
                let version = self
                    .service
                    .create_version(txn, number.clone(), schema)
                    .await?;

                if let Some(source_host) = link.host() {
                    if source_host.is_loopback(txn.host().into()) {
                        debug!("no need to replicate from {link}");
                    } else {
                        let source = link.append(number);

                        debug!(
                            "{} will replicate a new Service version {number} from {source}...",
                            txn.host()
                        );

                        version.replicate(txn, source).await?;
                    }
                }

                Ok(())
            })
        }))
    }
}

impl<'a> From<&'a Service> for ServiceHandler<'a> {
    fn from(service: &'a Service) -> Self {
        Self { service }
    }
}

struct VersionHandler<'a> {
    service: &'a Service,
    version: &'a Id,
}

impl<'a> VersionHandler<'a> {
    fn new(service: &'a Service, version: &'a Id) -> Self {
        Self { service, version }
    }
}

impl<'a> Handler<'a, State> for VersionHandler<'a> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if key.is_none() {
                    let version = self
                        .service
                        .get_version_proto(*txn.id(), self.version)
                        .await?;

                    Ok(version.into())
                } else {
                    let attr_name: Id =
                        key.try_cast_into(|v| TCError::unexpected(v, "a library attribute name"))?;

                    let version = self.version.as_str().parse()?;
                    let version = self.service.get_version(*txn.id(), &version).await?;

                    version
                        .get_attribute(&attr_name)
                        .cloned()
                        .map(State::from)
                        .ok_or_else(|| not_found!("service attribute {attr_name}"))
                }
            })
        }))
    }
}

struct VersionAttrHandler<'a> {
    service: &'a Service,
    version: &'a Id,
    path: &'a [PathSegment],
}

impl<'a> VersionAttrHandler<'a> {
    fn new(service: &'a Service, version: &'a Id, path: &'a [PathSegment]) -> Self {
        Self {
            service,
            version,
            path,
        }
    }
}

impl<'a> Handler<'a, State> for VersionAttrHandler<'a> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                let number = self.version.as_str().parse()?;
                let version = self.service.get_version(*txn.id(), &number).await?;
                version.get(txn, self.path, key).await
            })
        }))
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key, value| {
            Box::pin(async move {
                let number = self.version.as_str().parse()?;
                let version = self.service.get_version(*txn.id(), &number).await?;
                version.put(txn, self.path, key, value).await
            })
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, params| {
            Box::pin(async move {
                let number = self.version.as_str().parse()?;
                let version = self.service.get_version(*txn.id(), &number).await?;
                version.post(txn, self.path, params).await
            })
        }))
    }

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b, Txn>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                let number = self.version.as_str().parse()?;
                let version = self.service.get_version(*txn.id(), &number).await?;
                version.delete(txn, self.path, key).await
            })
        }))
    }
}

impl Route<State> for Service {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        if path.is_empty() {
            Some(Box::new(ServiceHandler::from(self)))
        } else if path.len() == 1 {
            Some(Box::new(VersionHandler::new(self, &path[0])))
        } else {
            Some(Box::new(VersionAttrHandler::new(
                self,
                &path[0],
                &path[1..],
            )))
        }
    }
}

impl Route<State> for Version {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        debug!("service::Version::route {}", TCPath::from(path));

        if path.is_empty() {
            return Some(Box::new(MethodNotAllowedHandler));
        }

        match self.get_attribute(&path[0]) {
            Some(attr) => match attr {
                Attr::Chain(chain) => chain.route(&path[1..]),
                Attr::Scalar(scalar) => route_attr(self, &path[0], scalar, &path[1..]),
            },
            None => {
                trace!("{:?} has no attr {}", self, &path[0]);
                None
            }
        }
    }
}
