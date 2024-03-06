use futures::TryFutureExt;
use safecast::TryCastInto;

use tc_error::{TCError, TCResult};
use tc_state::object::InstanceClass;
use tc_transact::public::{GetHandler, Handler, PutHandler, Route};
use tc_transact::Transaction;
use tc_value::Version as VersionNumber;
use tcgeneric::{Id, Map, PathSegment};

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
                let version_number: VersionNumber =
                    key.try_cast_into(|v| TCError::unexpected(v, "a semantic version number"))?;

                let version = self.class.get_version(*txn.id(), &version_number).await?;

                version.to_state(*txn.id()).await
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

struct ClassVersionHandler<'a> {
    class: &'a Class,
    version_id: &'a Id,
}

impl<'a> ClassVersionHandler<'a> {
    fn new(class: &'a Class, version_id: &'a Id) -> Self {
        Self { class, version_id }
    }
}

impl<'a> Handler<'a, State> for ClassVersionHandler<'a> {
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

impl Route<State> for Class {
    fn route<'a>(&'a self, path: &'a [PathSegment]) -> Option<Box<dyn Handler<'a, State> + 'a>> {
        if path.is_empty() {
            Some(Box::new(ClassHandler::from(self)))
        } else if path.len() == 1 {
            Some(Box::new(ClassVersionHandler::new(self, &path[0])))
        } else {
            None
        }
    }
}
