use std::ops::Deref;

use log::{debug, info};
use safecast::{TryCastFrom, TryCastInto};

use tc_error::*;
use tc_scalar::value::{Link, Version as VersionNumber};
use tc_scalar::{OpRef, Scalar, TCRef};
use tc_state::object::InstanceClass;
use tc_transact::public::{GetHandler, Handler, PostHandler, PutHandler, Route};
use tc_transact::{Gateway, Transaction, TxnId};
use tc_value::Value;
use tcgeneric::{Id, Map, PathSegment, TCPath, TCPathBuf};

use crate::cluster::dir::DirItem;
use crate::cluster::Library;
use crate::kernel::CLASS;
use crate::{State, Txn};

struct LibraryHandler<'a> {
    library: &'a Library,
}

impl<'a> Handler<'a, State> for LibraryHandler<'a> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if key.is_some() {
                    let number: VersionNumber =
                        key.try_cast_into(|v| TCError::unexpected(v, "a semantic version number"))?;

                    let version = self.library.get_version(*txn.id(), &number).await?;

                    Ok(State::Scalar(version.clone().into()))
                } else {
                    let version_numbers = self.library.list_versions(*txn.id()).await?;
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
                let number: VersionNumber =
                    key.try_cast_into(|v| TCError::unexpected(v, "a semantic version number"))?;

                let (link, schema) = expect_version(value)?;

                let mut classes = Map::<InstanceClass>::new();
                let mut version = Map::<Scalar>::new();

                fn is_dep(scalar: &Scalar) -> bool {
                    if let Scalar::Ref(tc_ref) = scalar {
                        if let TCRef::Op(op_ref) = &**tc_ref {
                            if let OpRef::Post(_) = op_ref {
                                return true;
                            }
                        }
                    }

                    false
                }

                for (name, state) in schema.into_iter() {
                    let scalar = Scalar::try_from(state)?;

                    if is_dep(&scalar) {
                        let class = scalar
                            .try_cast_into(|s| TCError::unexpected(s, "a class definition"))?;

                        classes.insert(name, class);
                    } else {
                        version.insert(name, scalar);
                    }
                }

                self.library.create_version(txn, number, version).await?;

                if !classes.is_empty() {
                    let mut class_path = TCPathBuf::from(CLASS);
                    class_path.extend(link.path()[1..].iter().cloned());
                    info!("installing a class set at {class_path}");
                    txn.put(class_path, number.clone(), classes).await?;
                }

                Ok(())
            })
        }))
    }
}

impl<'a> From<&'a Library> for LibraryHandler<'a> {
    fn from(library: &'a Library) -> Self {
        Self { library }
    }
}

struct LibraryVersionHandler<'a> {
    library: &'a Library,
    version: &'a PathSegment,
}

impl<'a> LibraryVersionHandler<'a> {
    fn new(library: &'a Library, version: &'a PathSegment) -> Self {
        Self { library, version }
    }
}

impl<'a> Handler<'a, State> for LibraryVersionHandler<'a> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                let number: VersionNumber = self.version.as_str().parse()?;

                let version = self.library.get_version(*txn.id(), &number).await?;

                if key.is_none() {
                    Ok(State::Scalar(version.clone().into()))
                } else {
                    let attr_name: Id =
                        key.try_cast_into(|v| TCError::unexpected(v, "a library attribute name"))?;

                    version
                        .get(&attr_name)
                        .cloned()
                        .map(State::Scalar)
                        .ok_or_else(|| not_found!("library attribute {attr_name}"))
                }
            })
        }))
    }
}

struct LibraryAttrHandler<'a> {
    library: &'a Library,
    path: &'a [PathSegment],
}

impl<'a> LibraryAttrHandler<'a> {
    fn new(library: &'a Library, path: &'a [PathSegment]) -> LibraryAttrHandler<'a> {
        Self { library, path }
    }

    async fn get_version(&self, txn_id: TxnId) -> TCResult<impl Deref<Target = Map<Scalar>>> {
        let number: VersionNumber = self.path[0].as_str().parse()?;
        self.library.get_version(txn_id, &number).await
    }
}

impl<'a> Handler<'a, State> for LibraryAttrHandler<'a> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(move |txn, key| {
            Box::pin(async move {
                let version = self.get_version(*txn.id()).await?;

                debug!(
                    "execute GET {}: {} from library version",
                    TCPath::from(&self.path[1..]),
                    key
                );

                tc_transact::public::Public::get(&*version, txn, &self.path[1..], key).await
            })
        }))
    }

    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(move |txn, params| {
            Box::pin(async move {
                let version = self.get_version(*txn.id()).await?;
                tc_transact::public::Public::post(&*version, txn, &self.path[1..], params).await
            })
        }))
    }
}

impl Route<State> for Library {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        debug!("Library::route {}", TCPath::from(path));

        if path.is_empty() {
            Some(Box::new(LibraryHandler::from(self)))
        } else if path.len() == 1 {
            Some(Box::new(LibraryVersionHandler::new(self, &path[0])))
        } else {
            Some(Box::new(LibraryAttrHandler::new(self, path)))
        }
    }
}

#[inline]
fn expect_version(version: State) -> TCResult<(Link, Map<Scalar>)> {
    InstanceClass::try_cast_from(version, |v| TCError::unexpected(v, "a Class"))
        .map(|class| class.into_inner())
}
