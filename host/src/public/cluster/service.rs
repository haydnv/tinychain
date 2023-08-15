use log::{debug, trace};
use safecast::TryCastFrom;

use tc_chain::CHAIN;
use tc_error::*;
use tc_scalar::{OpRef, OpRefType, Scalar, Subject, TCRef};
use tc_state::object::public::method::route_attr;
use tc_state::object::InstanceClass;
use tc_state::State;
use tc_transact::public::helpers::MethodNotAllowedHandler;
use tc_transact::public::{Handler, Public, Route};
use tc_transact::{RPCClient, Transaction};
use tc_value::{Link, Value, Version as VersionNumber};
use tcgeneric::{Id, Map, PathSegment, TCPath, TCPathBuf, Tuple};

use crate::cluster::{service, DirItem, Replica, Service};
use crate::kernel::CLASS;
use crate::public::cluster::dir::{expect_version, extract_classes};
use crate::txn::Txn;

use super::dir::DirHandler;
use super::{DeleteHandler, GetHandler, PostHandler, PutHandler};

impl Route<State> for service::Version {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        debug!("service::Version::route {}", TCPath::from(path));

        if path.is_empty() {
            return Some(Box::new(MethodNotAllowedHandler::from(self)));
        }

        match self.get_attribute(&path[0]) {
            Some(attr) => match attr {
                service::Attr::Chain(chain) => chain.route(&path[1..]),
                service::Attr::Scalar(scalar) => route_attr(self, &path[0], scalar, &path[1..]),
            },
            None => {
                trace!(
                    "{:?} has no attr {} (attrs are {})",
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

    fn create_version<'b>(self: Box<Self>) -> PutHandler<'a, 'b, Txn, State>
    where
        'b: 'a,
    {
        assert!(self.path.is_empty());

        Box::new(|txn, key, value| {
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

                if !classes.is_empty() && txn.is_leader(link.path()) {
                    let mut class_path = TCPathBuf::from(CLASS);
                    class_path.extend(link.path()[1..].iter().cloned());

                    debug!(
                        "creating new Class set version {} at {}...",
                        number, class_path
                    );

                    txn.put(class_path, number.clone(), classes).await?;
                }

                let schema = InstanceClass::extend(link.clone(), schema);
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

impl<'a> Handler<'a, State> for ServiceHandler<'a> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State>>
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
                let path = &self.path[1..];

                if path.len() == 2 && path[1] == CHAIN {
                    if let Some(service::Attr::Chain(chain)) = version.get_attribute(&path[0]) {
                        return Ok(State::Chain(chain.clone()));
                    }
                }

                version.get(txn, &self.path[1..], key).await
            })
        }))
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b, Txn, State>>
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

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, Txn, State>>
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

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b, Txn>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if self.path.len() == 0 {
                    return Err(not_implemented!("delete a version of a Service"));
                }

                let number = self.path[0].as_str().parse()?;
                let version = self.service.get_version(*txn.id(), &number).await?;
                version.delete(txn, &self.path[1..], key).await
            })
        }))
    }
}

impl Route<State> for Service {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        debug!("Service::route {}", TCPath::from(path));
        Some(Box::new(ServiceHandler::new(self, path)))
    }
}

impl<'a> Handler<'a, State> for DirHandler<'a, Service> {
    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key, value| {
            Box::pin(async move {
                debug!("create new Service {} in {:?}", key, self.dir);

                let name = PathSegment::try_cast_from(key, |v| {
                    TCError::unexpected(v, "a path segment for a Service directory entry")
                })?;

                let (link, service) = expect_version(value)?;
                let (version, classes) = extract_classes(service)?;

                if link.path().len() <= 1 {
                    return Err(bad_request!(
                        "cannot create a new cluster at {}",
                        link.path(),
                    ));
                }

                let mut class_path = TCPathBuf::from(CLASS);
                class_path.extend(link.path()[1..].iter().cloned());

                let class_link: Link = if let Some(host) = link.host() {
                    (host.clone(), class_path.clone()).into()
                } else {
                    class_path.clone().into()
                };

                let class_dir_path = TCPathBuf::from(class_path[..class_path.len() - 1].to_vec());

                let parent_dir_path = &link.path()[..link.path().len() - 1];

                if version.is_empty() && classes.is_empty() {
                    if txn.is_leader(parent_dir_path) {
                        txn.put(class_dir_path, name.clone(), class_link).await?;
                    }

                    self.create_item_or_dir::<Map<State>>(txn, link, name, None)
                        .await
                } else {
                    if txn.is_leader(parent_dir_path) {
                        let classes = vec![class_link.into(), classes.into()].into();
                        txn.put(class_dir_path, name.clone(), State::Tuple(classes))
                            .await?;
                    }

                    let version = InstanceClass::extend(link.clone(), version);

                    self.create_item_or_dir(txn, link, name, Some(version))
                        .await
                }
            })
        }))
    }
}
