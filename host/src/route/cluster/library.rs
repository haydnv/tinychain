use log::debug;
use safecast::TryCastFrom;

use tc_error::*;
use tc_transact::Transaction;
use tc_value::{Link, Value, Version as VersionNumber};
use tcgeneric::{Map, PathSegment, TCPath, TCPathBuf};

use crate::cluster::{library, DirItem, Library};
use crate::object::InstanceClass;
use crate::route::cluster::dir::{expect_version, extract_classes};
use crate::route::object::method::route_attr;
use crate::route::{DeleteHandler, GetHandler, Handler, PostHandler, Public, PutHandler, Route};
use crate::scalar::{OpRefType, Scalar};
use crate::state::State;
use crate::CLASS;

use super::dir::DirHandler;

impl Route for library::Version {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        assert!(!path.is_empty());

        let attr = self.get_attribute(&path[0])?;
        route_attr(self, &path[0], attr, &path[1..])
    }
}

struct LibraryHandler<'a> {
    lib: &'a Library,
    path: &'a [PathSegment],
}

impl<'a> LibraryHandler<'a> {
    fn new(lib: &'a Library, path: &'a [PathSegment]) -> Self {
        Self { lib, path }
    }
}

impl<'a> Handler<'a> for LibraryHandler<'a> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b>>
    where
        'b: 'a,
    {
        assert!(!self.path.is_empty());

        Some(Box::new(|txn, key| {
            Box::pin(async move {
                debug!(
                    "route GET {} to version {}",
                    TCPath::from(&self.path[1..]),
                    &self.path[0]
                );

                let number = self.path[0].as_str().parse()?;
                let version = self.lib.get_version(*txn.id(), &number).await?;
                version.get(txn, &self.path[1..], key).await
            })
        }))
    }

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key, value| {
            if self.path.is_empty() {
                Box::pin(async move {
                    debug!("create new Library version {}", key);

                    let number = VersionNumber::try_cast_from(key, |v| {
                        TCError::bad_request("invalid version number", v)
                    })?;

                    let (link, version) = expect_version(value)?;
                    let (version, classes) = extract_classes(version)?;

                    if !classes.is_empty() && txn.is_leader(link.path()) {
                        let mut class_path = TCPathBuf::from(CLASS);
                        class_path.extend(link.path()[1..].iter().cloned());

                        txn.put(class_path.into(), number.clone().into(), classes.into())
                            .await?;
                    }

                    self.lib.create_version(txn, number, version).await?;

                    Ok(())
                })
            } else {
                Box::pin(async move {
                    let number = self.path[0].as_str().parse()?;
                    let version = self.lib.get_version(*txn.id(), &number).await?;
                    version.put(txn, &self.path[1..], key, value).await
                })
            }
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, params| {
            Box::pin(async move {
                let number = self.path[0].as_str().parse()?;
                let version = self.lib.get_version(*txn.id(), &number).await?;
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
                let number = self.path[0].as_str().parse()?;
                let version = self.lib.get_version(*txn.id(), &number).await?;
                version.delete(txn, &self.path[1..], key).await
            })
        }))
    }
}

impl Route for Library {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a> + 'a>> {
        Some(Box::new(LibraryHandler::new(self, path)))
    }
}

impl<'a> Handler<'a> for DirHandler<'a, Library> {
    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key, value| {
            Box::pin(async move {
                debug!("create new Library {} in {}", key, self.dir);

                let name = PathSegment::try_cast_from(key, |v| {
                    TCError::bad_request("invalid path segment for Library directory entry", v)
                })?;

                let (link, lib) = expect_version(value)?;
                let (version, classes) = extract_classes(lib)?;

                if link.path().len() <= 1 {
                    return Err(TCError::bad_request(
                        "cannot create a new cluster at",
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
                        txn.put(
                            class_dir_path.into(),
                            name.clone().into(),
                            class_link.into(),
                        )
                        .await?;
                    }

                    self.create_item_or_dir::<Map<Scalar>>(txn, link, name, None)
                        .await
                } else {
                    if txn.is_leader(parent_dir_path) {
                        txn.put(
                            class_dir_path.into(),
                            name.clone().into(),
                            State::Tuple((class_link.into(), classes.into()).into()),
                        )
                        .await?;
                    }

                    let version = InstanceClass::anonymous(Some(link.clone()), version);

                    self.create_item_or_dir(txn, link, name, Some(version))
                        .await
                }
            })
        }))
    }
}
