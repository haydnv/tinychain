use log::debug;
use safecast::TryCastInto;

use tc_error::TCError;
use tc_transact::Transaction;
use tc_value::Value;
use tcgeneric::{Map, PathSegment, TCPath};

use crate::cluster::{service, Service};
use crate::route::{DeleteHandler, GetHandler, Handler, PostHandler, Public, PutHandler, Route};
use crate::scalar::OpRefType;
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

                Err(TCError::not_implemented("install new Service"))
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
