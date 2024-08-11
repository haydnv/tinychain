use log::debug;
use safecast::{TryCastFrom, TryCastInto};

use tc_error::*;
use tc_scalar::{OpRef, Scalar, Subject, TCRef};
use tc_state::object::InstanceClass;
use tc_transact::public::{
    DeleteHandler, GetHandler, Handler, PostHandler, Public, PutHandler, Route,
};
use tc_transact::{Replicate, Transaction, TxnId};
use tc_value::{Value, Version as VersionNumber};
use tcgeneric::{Id, Map, PathSegment};

use crate::cluster::dir::DirItem;
use crate::cluster::service::{Attr, Service};
use crate::{State, Txn};

impl Route<State> for Attr {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        match self {
            Self::Chain(chain) => chain.route(path),
            Self::Scalar(scalar) => scalar.route(path),
        }
    }
}

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

                let mut classes = Map::new();
                let mut schema = Map::new();

                for (name, scalar) in version {
                    match scalar {
                        Scalar::Ref(tc_ref) => match *tc_ref {
                            TCRef::Op(OpRef::Post((Subject::Link(classpath), proto)))
                                if !proto.is_empty() =>
                            {
                                let class = InstanceClass::extend(classpath, proto);
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

                let schema = InstanceClass::extend(link.clone(), schema);
                let version = self
                    .service
                    .create_version(txn, number.clone(), schema)
                    .await?;

                if link.host() == Some(txn.host()) {
                    debug!("no need to replicate from {link}");
                } else {
                    let source = link.append(number);
                    debug!("replicating new service version from {source}...");
                    version.replicate(txn, source).await?;
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

struct ServiceAttrHandler<'a> {
    service: &'a Service,
    version: &'a Id,
    attr_name: &'a Id,
    path: &'a [PathSegment],
}

impl<'a> ServiceAttrHandler<'a> {
    fn new(
        service: &'a Service,
        version: &'a Id,
        attr_name: &'a Id,
        path: &'a [PathSegment],
    ) -> Self {
        Self {
            service,
            version,
            attr_name,
            path,
        }
    }

    async fn get_attr(&self, txn_id: TxnId) -> TCResult<Attr> {
        let number = self.version.as_str().parse()?;
        let version = self.service.get_version(txn_id, &number).await?;

        version
            .get_attribute(self.attr_name)
            .cloned()
            .ok_or_else(|| not_found!("Service attribute {}", self.attr_name))
    }
}

impl<'a> Handler<'a, State> for ServiceAttrHandler<'a> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                let attr = self.get_attr(*txn.id()).await?;
                attr.get(txn, self.path, key).await
            })
        }))
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key, value| {
            Box::pin(async move {
                let attr = self.get_attr(*txn.id()).await?;
                attr.put(txn, self.path, key, value).await
            })
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, params| {
            Box::pin(async move {
                let attr = self.get_attr(*txn.id()).await?;
                attr.post(txn, self.path, params).await
            })
        }))
    }

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b, Txn>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                let attr = self.get_attr(*txn.id()).await?;
                attr.delete(txn, self.path, key).await
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
            Some(Box::new(ServiceAttrHandler::new(
                self,
                &path[0],
                &path[1],
                &path[2..],
            )))
        }
    }
}
