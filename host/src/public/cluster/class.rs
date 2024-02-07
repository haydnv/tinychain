use log::debug;
use safecast::{CastInto, TryCastFrom, TryCastInto};

use tc_error::*;
use tc_scalar::OpRefType;
use tc_state::object::{InstanceClass, Object};
use tc_transact::public::{
    DeleteHandler, GetHandler, Handler, PostHandler, Public, PutHandler, Route,
};
use tc_transact::Transaction;
use tc_value::Link;
use tcgeneric::{Map, PathSegment, TCPath};

use crate::cluster::{class, Class, DirItem};
use crate::state::State;
use crate::txn::Txn;

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

impl<'a> Handler<'a, State> for ClassVersionHandler<'a> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if self.path.is_empty() {
                    let name = key.try_cast_into(|v| TCError::unexpected(v, "a Class name"))?;
                    let class = self.class.get_class(*txn.id(), &name).await?;
                    Ok(State::Object(class.into()))
                } else {
                    let class = self.class.get_class(*txn.id(), &self.path[0]).await?;
                    class.get(txn, &self.path[1..], key).await
                }
            })
        }))
    }
}

impl Route<State> for class::Version {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
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

impl<'a> Handler<'a, State> for ClassHandler<'a> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State>>
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

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        assert!(self.path.is_empty());

        Some(Box::new(|txn, key, value| {
            Box::pin(async move {
                if self.path.is_empty() {
                    let number =
                        key.try_cast_into(|v| TCError::unexpected(v, "a version number"))?;

                    let version =
                        value.try_into_map(|s| TCError::unexpected(s, "a Map of Classes"))?;

                    let version = version
                        .into_iter()
                        .map(|(name, class)| {
                            InstanceClass::try_cast_from(class, |s| {
                                TCError::unexpected(s, "a Class")
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

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, Txn, State>>
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

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b, Txn>>
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

impl Route<State> for Class {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        Some(Box::new(ClassHandler::new(self, path)))
    }
}

impl<'a> Handler<'a, State> for DirHandler<'a, Class> {
    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key, value| {
            Box::pin(async move {
                debug!("create new Class directory entry at {key} with classes {value:?}");

                let name = PathSegment::try_cast_from(key, |v| {
                    TCError::unexpected(v, "a path segment for a Class directory entry")
                })?;

                let (link, classes): (Link, Option<Map<InstanceClass>>) =
                    if Link::can_cast_from(&value) {
                        let link = value.opt_cast_into().expect("class dir link");
                        (link, None)
                    } else {
                        // TODO: is there a better way to implement this?
                        let (link, classes) = match value {
                            State::Object(Object::Class(class)) => Ok(class.cast_into()),
                            State::Scalar(scalar) => scalar.try_cast_into(|s| {
                                bad_request!(
                                    "invalid class set {name}: {s:?}, expected (Link, (Class...))"
                                )
                            }),
                            State::Tuple(tuple) => tuple.try_cast_into(|s| {
                                bad_request!(
                                    "invalid class set {name}: {s:?}, expected (Link, (Class...))"
                                )
                            }),
                            other => Err(bad_request!(
                                "invalid class set {name}: {other:?}, expected (Link, (Class...))"
                            )),
                        }?;

                        let classes = classes
                            .into_iter()
                            .map(|(name, class)| {
                                InstanceClass::try_cast_from(class, |s| {
                                    TCError::unexpected(s, "a Class definition")
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
