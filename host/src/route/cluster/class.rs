use log::debug;
use safecast::{TryCastFrom, TryCastInto};

use tc_error::*;
use tc_transact::Transaction;
use tc_value::Link;

use crate::cluster::{class, Class, DirItem};
use crate::object::InstanceClass;
use crate::route::*;
use crate::scalar::OpRefType;
use crate::state::State;

use super::dir::DirHandler;

struct ClassVersionHandler<'a> {
    class: &'a class::Version,
    path: &'a [PathSegment],
}

impl<'a> ClassVersionHandler<'a> {
    fn new(class: &'a class::Version, path: &'a [PathSegment]) -> Self {
        Self { class, path }
    }
}

impl<'a> Handler<'a> for ClassVersionHandler<'a> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if self.path.is_empty() {
                    let name =
                        key.try_cast_into(|v| TCError::bad_request("invalid class name", v))?;

                    let class = self.class.get_class(*txn.id(), &name).await?;
                    Ok(State::Object(class.clone().into()))
                } else {
                    let class = self.class.get_class(*txn.id(), &self.path[0]).await?;
                    class.get(txn, &self.path[1..], key).await
                }
            })
        }))
    }
}

impl Route for class::Version {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        Some(Box::new(ClassVersionHandler::new(self, path)))
    }
}

struct ClassHandler<'a> {
    class: &'a Class,
    path: &'a [PathSegment],
}

impl<'a> ClassHandler<'a> {
    fn new(class: &'a Class, path: &'a [PathSegment]) -> Self {
        Self { class, path }
    }
}

impl<'a> Handler<'a> for ClassHandler<'a> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                let number = self.path[0].as_str().parse()?;
                let version = self.class.get_version(*txn.id(), &number).await?;
                version.get(txn, &self.path[1..], key).await
            })
        }))
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b>>
    where
        'b: 'a,
    {
        assert!(self.path.is_empty());

        Some(Box::new(|txn, key, value| {
            Box::pin(async move {
                if self.path.is_empty() {
                    let number =
                        key.try_cast_into(|v| TCError::bad_request("invalid version number", v))?;

                    let version = value.try_into_map(|s| {
                        TCError::bad_request("expected a Map of Classes but found", s)
                    })?;

                    let version = version
                        .into_iter()
                        .map(|(name, class)| {
                            InstanceClass::try_cast_from(class, |s| {
                                TCError::bad_request("expected a Class but found", s)
                            })
                            .map(|class| (name, class))
                        })
                        .collect::<TCResult<Map<InstanceClass>>>()?;

                    self.class.create_version(txn, number, version).await?;

                    Ok(())
                } else {
                    let number = self.path[0].as_str().parse()?;
                    let version = self.class.get_version(*txn.id(), &number).await?;
                    version.put(txn, &self.path[1..], key, value).await
                }
            })
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, params| {
            Box::pin(async move {
                if self.path.len() < 2 {
                    return Err(TCError::method_not_allowed(
                        OpRefType::Post,
                        self.class,
                        TCPath::from(self.path),
                    ));
                }

                let number = self.path[0].as_str().parse()?;
                let version = self.class.get_version(*txn.id(), &number).await?;
                let class = version.get_class(*txn.id(), &self.path[1]).await?;
                class.post(txn, &self.path[2..], params).await
            })
        }))
    }

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                let number = self.path[0].as_str().parse()?;
                let version = self.class.get_version(*txn.id(), &number).await?;
                version.delete(txn, &self.path[1..], key).await
            })
        }))
    }
}

impl Route for Class {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        Some(Box::new(ClassHandler::new(self, path)))
    }
}

impl<'a> Handler<'a> for DirHandler<'a, Class> {
    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key, value| {
            Box::pin(async move {
                debug!("create new Class directory entry at {}", key);

                let name = PathSegment::try_cast_from(key, |v| {
                    TCError::bad_request("invalid path segment for Class directory entry", v)
                })?;

                let (link, classes): (Link, Option<Map<InstanceClass>>) =
                    if Link::can_cast_from(&value) {
                        let link = value.opt_cast_into().expect("class dir link");
                        (link, None)
                    } else {
                        let (link, classes): (Link, Map<State>) = value.try_cast_into(|s| {
                            TCError::bad_request("expected a tuple (Link, (Class...)) but found", s)
                        })?;

                        let classes = classes
                            .into_iter()
                            .map(|(name, class)| {
                                InstanceClass::try_cast_from(class, |s| {
                                    TCError::bad_request("invalid Class definition", s)
                                })
                                .map(|class| (name, class))
                            })
                            .collect::<TCResult<_>>()?;

                        (link, Some(classes))
                    };

                self.create_item_or_dir(txn, link, name, classes).await
            })
        }))
    }
}
