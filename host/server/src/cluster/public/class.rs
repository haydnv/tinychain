use futures::TryFutureExt;
use log::{debug, trace};
use safecast::TryCastInto;

use tc_error::{not_found, TCError, TCResult};
use tc_state::object::InstanceClass;
use tc_transact::public::{GetHandler, Handler, PostHandler, PutHandler, Route};
use tc_transact::Transaction;
use tc_value::{Value, Version as VersionNumber};
use tcgeneric::{Id, Map, PathSegment, TCPath};

use crate::cluster::dir::DirItem;
use crate::cluster::Class;
use crate::{State, Txn};

struct ClassHandler<'a> {
    class: &'a Class,
}

impl<'a> Handler<'a, State> for ClassHandler<'a> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                if key.is_some() {
                    let version_number: VersionNumber =
                        key.try_cast_into(|v| TCError::unexpected(v, "a semantic version number"))?;

                    let version = self.class.get_version(*txn.id(), &version_number).await?;

                    version.to_state(*txn.id()).await
                } else {
                    let versions = self.class.list_versions(*txn.id()).await?;

                    let version_ids = versions
                        .map(|(version_number, _)| version_number)
                        .map(Id::from)
                        .collect();

                    Ok(Value::Tuple(version_ids).into())
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
                let version_number: VersionNumber =
                    key.try_cast_into(|v| TCError::unexpected(v, "a semantic version number"))?;

                let schema =
                    value.try_into_map(|s| TCError::unexpected(s, "a set of named classes"))?;

                let schema = schema
                    .into_iter()
                    .map(|(name, class)| {
                        class
                            .try_cast_into(|c| TCError::unexpected(c, "a class definition"))
                            .map(|class: InstanceClass| (name, class))
                    })
                    .collect::<TCResult<Map<InstanceClass>>>()?;

                debug!("create new class set version {version_number}: {schema:?}");

                self.class
                    .create_version(txn, version_number, schema)
                    .await?;

                Ok(())
            })
        }))
    }
}

impl<'a> From<&'a Class> for ClassHandler<'a> {
    fn from(class: &'a Class) -> Self {
        Self { class }
    }
}

struct VersionHandler<'a> {
    class: &'a Class,
    version_id: &'a Id,
}

impl<'a> VersionHandler<'a> {
    fn new(class: &'a Class, version_id: &'a Id) -> Self {
        Self { class, version_id }
    }
}

impl<'a> Handler<'a, State> for VersionHandler<'a> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                let version_number = self.version_id.as_str().parse()?;
                let version = self.class.get_version(*txn.id(), &version_number).await?;

                if key.is_none() {
                    version.to_state(*txn.id()).await
                } else {
                    let class_name =
                        key.try_cast_into(|v| TCError::unexpected(v, "a class name"))?;

                    version
                        .get_class(*txn.id(), &class_name)
                        .map_ok(State::from)
                        .await
                }
            })
        }))
    }
}

struct VersionClassHandler<'a> {
    class: &'a Class,
    version_id: &'a Id,
    name: &'a Id,
}

impl<'a> VersionClassHandler<'a> {
    fn new(class: &'a Class, version_id: &'a Id, name: &'a Id) -> Self {
        Self {
            class,
            version_id,
            name,
        }
    }
}

impl<'a> Handler<'a, State> for VersionClassHandler<'a> {
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State>>
    where
        'b: 'a,
    {
        Some(Box::new(|txn, key| {
            Box::pin(async move {
                let version_number = self.version_id.as_str().parse()?;
                let version = self.class.get_version(*txn.id(), &version_number).await?;
                let class = version.get_class(*txn.id(), self.name).await?;

                if key.is_none() {
                    Ok(class.into())
                } else {
                    trace!("get attr {} of {:?}", self.name, self.class);

                    let attr_name: Id =
                        key.try_cast_into(|v| TCError::unexpected(v, "a class attribute name"))?;

                    let attr = class
                        .proto()
                        .get(&attr_name)
                        .ok_or_else(|| not_found!("{attr_name}"))?;

                    Ok(attr.clone().into())
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
                trace!("check class set version {}...", self.version_id);
                let number = self.version_id.as_str().parse()?;
                let version = self.class.get_version(*txn.id(), &number).await?;

                trace!(
                    "check for class {} in version {}...",
                    self.name,
                    self.version_id
                );

                let class = version.get_class(*txn.id(), &self.name).await?;

                trace!("construct a new instance of {class:?}");
                tc_transact::public::Public::post(&class, txn, &[], params).await
            })
        }))
    }
}

impl Route<State> for Class {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        debug!("Class::route {}", TCPath::from(path));

        if path.is_empty() {
            Some(Box::new(ClassHandler::from(self)))
        } else if path.len() == 1 {
            Some(Box::new(VersionHandler::new(self, &path[0])))
        } else if path.len() == 2 {
            Some(Box::new(VersionClassHandler::new(self, &path[0], &path[1])))
        } else {
            None
        }
    }
}
