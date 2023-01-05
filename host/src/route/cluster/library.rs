use log::debug;
use safecast::{CastInto, TryCastFrom, TryCastInto};

use tc_error::*;
use tc_transact::Transaction;
use tc_value::{Link, Value, Version as VersionNumber};
use tcgeneric::{Id, Map, PathSegment, TCPath, TCPathBuf};

use crate::cluster::{library, DirItem, Library};
use crate::object::InstanceClass;
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

    fn lib_classes(mut lib: Map<Scalar>) -> TCResult<(Map<Scalar>, Map<InstanceClass>)> {
        let deps = lib
            .iter()
            .filter(|(_, scalar)| scalar.is_ref())
            .map(|(name, _)| name.clone())
            .collect::<Vec<Id>>();

        let classes = deps
            .into_iter()
            .filter_map(|name| lib.remove(&name).map(|dep| (name, dep)))
            .map(|(name, dep)| {
                InstanceClass::try_cast_from(dep, |s| {
                    TCError::bad_request("unable to resolve Library dependency", s)
                })
                .map(|class| (name, class))
            })
            .collect::<TCResult<Map<InstanceClass>>>()?;

        Ok((lib, classes))
    }

    fn lib_version(version: State) -> TCResult<(Link, Map<Scalar>)> {
        let class =
            InstanceClass::try_cast_from(version, |v| TCError::bad_request("invalid Class", v))?;

        let (link, version) = class.into_inner();
        let link =
            link.ok_or_else(|| TCError::bad_request("missing cluster link for", &version))?;

        Ok((link, version))
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
                    debug!("{} <- {}: {}", self.lib, key, value);

                    let number = VersionNumber::try_cast_from(key, |v| {
                        TCError::bad_request("invalid version number", v)
                    })?;

                    let (link, version) = Self::lib_version(value)?;
                    let (version, classes) = Self::lib_classes(version)?;

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

                let (link, lib) = LibraryHandler::lib_version(value)?;

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

                if lib.is_empty() {
                    if txn.is_leader(parent_dir_path) {
                        txn.put(
                            class_dir_path.into(),
                            name.clone().into(),
                            class_link.into(),
                        )
                        .await?;
                    }

                    return self
                        .create_item_or_dir::<Map<Scalar>>(txn, link, name, None)
                        .await;
                }

                let (lib, classes) = LibraryHandler::lib_classes(lib)?;

                if !classes.is_empty() && txn.is_leader(parent_dir_path) {
                    txn.put(
                        class_dir_path.into(),
                        name.clone().into(),
                        (class_link, classes).cast_into(),
                    )
                    .await?;
                }

                let version = InstanceClass::anonymous(Some(link.clone()), lib);

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
