use log::{debug, trace};
use safecast::TryCastFrom;

use tc_error::*;
use tc_scalar::Scalar;
use tc_state::object::InstanceClass;
use tc_transact::public::{
    DeleteHandler, GetHandler, Handler, PostHandler, Public, PutHandler, Route,
};
use tc_transact::{Gateway, Transaction};
use tc_value::{Link, Version as VersionNumber};
use tcgeneric::{Map, PathSegment, TCPath, TCPathBuf};

use crate::cluster::{DirItem, Library};
use crate::kernel::CLASS;
use crate::state::State;
use crate::txn::Txn;

use super::authorize_install;
use super::dir::{expect_version, extract_classes, DirHandler};

struct LibraryHandler<'a> {
    lib: &'a Library,
    path: &'a [PathSegment],
}

impl<'a> LibraryHandler<'a> {
    fn new(lib: &'a Library, path: &'a [PathSegment]) -> Self {
        Self { lib, path }
    }
}

impl<'a> Handler<'a, State> for LibraryHandler<'a> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State>>
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

    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key, value| {
            if self.path.is_empty() {
                Box::pin(async move {
                    debug!("create new Library version {}", key);

                    let number = VersionNumber::try_cast_from(key, |v| {
                        TCError::unexpected(v, "a version number")
                    })?;

                    let (link, version) = expect_version(value)?;

                    {
                        let (host, mut path) = link.clone().into_inner();

                        let name = path.pop().ok_or_else(|| {
                            bad_request!("cluster link {} is missing a path", &link)
                        })?;

                        let parent = (host, path).into();
                        let entry_path = [name, number.clone().into()].into_iter().collect();
                        authorize_install(txn, &parent, &entry_path)?;
                    }

                    let (version, classes) = extract_classes(version)?;

                    if !classes.is_empty() && txn.is_leader(link.path()) {
                        debug!(
                            "library defines {} classes to host under /class",
                            classes.len()
                        );

                        let mut class_path =
                            TCPathBuf::with_capacity(link.path().len()).append(CLASS);
                        class_path.extend(link.path()[1..].iter().cloned());

                        txn.put(class_path, number.clone(), classes).await?;
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

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, Txn, State>>
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

    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b, Txn>>
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

impl Route<State> for Library {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        Some(Box::new(LibraryHandler::new(self, path)))
    }
}

impl<'a> Handler<'a, State> for DirHandler<'a, Library> {
    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key, value| {
            Box::pin(async move {
                debug!("create new Library {} in {:?}", key, self.dir);

                let name = PathSegment::try_cast_from(key, |v| {
                    TCError::unexpected(v, "a path segment for a Library directory entry")
                })?;

                let (link, lib) = expect_version(value)?;

                {
                    let mut parent = link.clone();
                    if parent.path_mut().pop().as_ref() != Some(&name) {
                        return Err(bad_request!("invalid link for {}: {}", name, parent));
                    }

                    authorize_install(txn, &parent, &TCPathBuf::from(name.clone()))?;
                }

                let (version, classes) = extract_classes(lib)?;

                if link.path().len() <= 1 {
                    return Err(bad_request!(
                        "cannot create a new cluster at {}",
                        link.path(),
                    ));
                }

                let mut class_path = TCPathBuf::with_capacity(link.path().len()).append(CLASS);
                class_path.extend(link.path()[1..].iter().cloned());

                let class_link: Link = if let Some(host) = link.host() {
                    (host.clone(), class_path.clone()).into()
                } else {
                    class_path.clone().into()
                };

                let class_dir_path = TCPathBuf::from_slice(&class_path[..class_path.len() - 1]);

                let parent_dir_path = &link.path()[..link.path().len() - 1];

                if version.is_empty() && classes.is_empty() {
                    if txn.is_leader(parent_dir_path) {
                        txn.put(class_dir_path, name.clone(), class_link).await?;
                    }

                    self.create_item_or_dir::<Map<Scalar>>(txn, link, name, None)
                        .await
                } else {
                    if txn.is_leader(parent_dir_path) {
                        debug!("library depends on {} classes", classes.len());
                        trace!("replicate classes {classes:?} at {name} in {class_dir_path}...");

                        txn.put(
                            class_dir_path,
                            name.clone(),
                            State::Tuple((class_link.into(), classes.into()).into()),
                        )
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
