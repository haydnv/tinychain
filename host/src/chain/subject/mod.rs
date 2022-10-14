//! The [`Subject`] of a `Chain`

use std::convert::{TryFrom, TryInto};
use std::fmt;

use async_hash::Hash;
use async_trait::async_trait;
use destream::de;
use futures::future::{join_all, try_join_all, TryFutureExt};
use log::{debug, trace};
use sha2::digest::{Digest, Output};
use sha2::Sha256;

use tc_error::*;
use tc_transact::fs::{Dir, DirRead, DirWrite};
use tc_transact::{IntoView, Transact, Transaction, TxnId};
use tcgeneric::*;

use crate::collection::Collection;
use crate::fs;
use crate::state::{State, StateView};
use crate::txn::Txn;

use super::{CollectionSchema, Schema};

pub use collection::SubjectCollection;
pub use map::SubjectMap;

mod collection;
mod map;

const DYNAMIC: Label = label("dynamic");
const SUBJECT: Label = label("subject");

/// The state whose transactional integrity is protected by a `Chain`
#[derive(Clone)]
pub enum Subject {
    Collection(SubjectCollection),
    Dynamic(SubjectMap),
    Map(Map<Subject>),
    Tuple(Tuple<Subject>),
}

impl Subject {
    fn from_state(state: State) -> TCResult<Subject> {
        match state {
            State::Collection(collection) => {
                SubjectCollection::from_collection(collection).map(Self::Collection)
            }

            State::Map(map) => {
                let subject = map
                    .into_iter()
                    .map(|(name, state)| Subject::from_state(state).map(|subject| (name, subject)))
                    .collect::<TCResult<Map<Subject>>>()?;

                Ok(Subject::Map(subject))
            }
            State::Tuple(tuple) => {
                let subject = tuple
                    .into_iter()
                    .map(Subject::from_state)
                    .collect::<TCResult<Tuple<Subject>>>()?;

                Ok(Subject::Tuple(subject))
            }

            other => Err(TCError::bad_request(
                "Chain expected a Collection, Map, or Tuple, not",
                other,
            )),
        }
    }

    /// Create a new `Subject` with the given `Schema`.
    pub fn create(schema: Schema, dir: &fs::Dir, txn_id: TxnId) -> TCBoxTryFuture<Self> {
        Box::pin(async move {
            match schema {
                Schema::Collection(schema) => {
                    SubjectCollection::create(schema, dir, txn_id, SUBJECT.into())
                        .map_ok(Self::Collection)
                        .await
                }
                Schema::Tuple(schema) => {
                    let mut container = dir.write(txn_id).await?;
                    let mut subjects = Vec::with_capacity(schema.len());

                    for (i, schema) in schema.into_iter().enumerate() {
                        let dir = container.create_dir(i.into())?;
                        let subject = Self::create(schema, &dir, txn_id).await?;
                        subjects.push(subject);
                    }

                    Ok(Self::Tuple(subjects.into()))
                }
                Schema::Map(schema) if schema.is_empty() => {
                    SubjectMap::create(dir.clone(), txn_id)
                        .map_ok(Self::Dynamic)
                        .await
                }
                Schema::Map(schema) => {
                    let mut container = dir.write(txn_id).await?;
                    let mut subjects = Map::new();

                    for (name, schema) in schema.into_iter() {
                        let dir = container.create_dir(name.clone())?;
                        let subject = Self::create(schema, &dir, txn_id).await?;
                        subjects.insert(name, subject);
                    }

                    Ok(Self::Map(subjects))
                }
            }
        })
    }

    pub(super) fn load<'a>(
        txn: &'a Txn,
        schema: Schema,
        dir: &'a fs::Dir,
    ) -> TCBoxTryFuture<'a, Self> {
        Box::pin(async move {
            debug!("Subject::load");

            let txn_id = *txn.id();

            match schema {
                Schema::Collection(schema) => {
                    trace!("load collection {}", schema);

                    SubjectCollection::load(txn, schema, dir, SUBJECT.into())
                        .map_ok(Self::Collection)
                        .await
                }

                Schema::Map(schema) if schema.is_empty() => {
                    trace!("load map {}", schema);

                    let mut container = dir.write(txn_id).await?;
                    if let Some(dir) = container.get_dir(&DYNAMIC.into())? {
                        SubjectMap::load(txn, dir).map_ok(Self::Dynamic).await
                    } else {
                        let dir = container.create_dir(DYNAMIC.into())?;
                        SubjectMap::create(dir, txn_id).map_ok(Self::Dynamic).await
                    }
                }
                Schema::Map(schema) => {
                    trace!("load map {}", schema);

                    let mut container = dir.write(txn_id).await?;
                    let mut subjects = Map::new();

                    for (name, schema) in schema.into_iter() {
                        let subject = if let Some(dir) = container.get_dir(&name)? {
                            Self::load(txn, schema, &dir).await
                        } else {
                            let dir = container.create_dir(name.clone())?;
                            Self::create(schema, &dir, txn_id).await
                        }?;

                        subjects.insert(name, subject);
                    }

                    Ok(Self::Map(subjects))
                }

                Schema::Tuple(schema) => {
                    trace!("load tuple {}", schema);

                    let mut container = dir.write(txn_id).await?;
                    let mut subjects = Vec::with_capacity(schema.len());

                    for (i, schema) in schema.into_iter().enumerate() {
                        let subject = if let Some(dir) = container.get_dir(&i.into())? {
                            Self::load(txn, schema, &dir).await
                        } else {
                            let dir = container.create_dir(i.into())?;
                            Self::create(schema, &dir, txn_id).await
                        }?;

                        subjects.push(subject);
                    }

                    Ok(Self::Tuple(subjects.into()))
                }
            }
        })
    }

    pub(super) fn restore<'a>(&'a self, txn: &'a Txn, backup: State) -> TCBoxTryFuture<()> {
        Box::pin(async move {
            match self {
                Self::Collection(subject) => {
                    let backup = backup.try_into()?;
                    subject.restore(txn, backup).await
                }

                Self::Dynamic(subject) => match backup {
                    State::Map(collections) => {
                        let backups = collections
                            .into_iter()
                            .map(|(id, c)| Collection::try_from(c).map(|c| (id, c)))
                            .collect::<TCResult<Map<Collection>>>()?;

                        subject.restore(txn, backups).await
                    }
                    other => Err(TCError::bad_request(
                        "cannot restore dynamic Chain from",
                        other,
                    )),
                },

                Self::Map(map) => match backup {
                    State::Map(mut backups) => {
                        let backups = map
                            .iter()
                            .map(|(name, subject)| {
                                backups
                                    .remove(name)
                                    .ok_or_else(|| {
                                        TCError::bad_request(
                                            "backup not found for Chain subject",
                                            name,
                                        )
                                    })
                                    .map(|backup| (subject, backup))
                            })
                            .collect::<TCResult<Vec<(&Subject, State)>>>()?;

                        let restores = backups
                            .into_iter()
                            .map(|(subject, backup)| subject.restore(txn, backup));

                        try_join_all(restores).await?;

                        Ok(())
                    }
                    backup => Err(TCError::unsupported(format!(
                        "invalid backup for schema {}: {}",
                        map, backup
                    ))),
                },

                Self::Tuple(tuple) => match backup {
                    State::Tuple(backup) if backup.len() == tuple.len() => {
                        let restores =
                            tuple
                                .iter()
                                .zip(backup)
                                .map(|(subject, backup)| async move {
                                    subject.restore(txn, backup).await
                                });

                        try_join_all(restores).await?;
                        Ok(())
                    }
                    State::Tuple(_) => Err(TCError::bad_request(
                        "backup has the wrong number of subjects for schema",
                        tuple,
                    )),
                    backup => Err(TCError::unsupported(format!(
                        "invalid backup for schema {}: {}",
                        tuple, backup
                    ))),
                },
            }
        })
    }

    pub fn into_state<'a>(self, txn_id: TxnId) -> TCBoxTryFuture<'a, State> {
        Box::pin(async move {
            match self {
                Self::Collection(subject) => subject.into_state(txn_id).await,
                Self::Dynamic(subject) => subject.into_state(txn_id).await,
                Self::Map(subject) => {
                    let mut state = Map::new();
                    for (id, subject) in subject.into_iter() {
                        let subject = subject.into_state(txn_id).await?;
                        state.insert(id, subject);
                    }
                    Ok(State::Map(state))
                }
                Self::Tuple(subject) => {
                    let mut state = Vec::with_capacity(subject.len());
                    for subject in subject.into_iter() {
                        let subject = subject.into_state(txn_id).await?;
                        state.push(subject);
                    }
                    Ok(State::Tuple(state.into()))
                }
            }
        })
    }

    pub fn hash<'a>(self, txn: Txn) -> TCBoxTryFuture<'a, Output<Sha256>> {
        Box::pin(async move {
            // TODO: should this be consolidated with Collection::hash?
            match self {
                Self::Collection(subject) => subject.hash(txn).await,

                Self::Dynamic(subject) => subject.hash(txn).await,

                Self::Map(map) => {
                    let mut hasher = Sha256::default();
                    for (id, subject) in map {
                        let subject = subject.hash(txn.clone()).await?;

                        let mut inner_hasher = Sha256::default();
                        inner_hasher.update(&Hash::<Sha256>::hash(id));
                        inner_hasher.update(&subject);
                        hasher.update(&inner_hasher.finalize());
                    }

                    Ok(hasher.finalize())
                }

                Self::Tuple(tuple) => {
                    let mut hasher = Sha256::default();
                    for subject in tuple {
                        let subject = subject.hash(txn.clone()).await?;
                        hasher.update(&subject);
                    }
                    Ok(hasher.finalize())
                }
            }
        })
    }
}

#[async_trait]
impl Transact for Subject {
    type Commit = ();

    async fn commit(&self, txn_id: &TxnId) {
        debug!("commit chain subject");

        match self {
            Self::Collection(subject) => subject.commit(txn_id).await,
            Self::Dynamic(subject) => subject.commit(txn_id).await,
            Self::Map(map) => {
                join_all(
                    map.iter()
                        .map(|(_, subject)| async move { subject.commit(txn_id).await }),
                )
                .await;
            }
            Self::Tuple(tuple) => {
                join_all(
                    tuple
                        .iter()
                        .map(|subject| async move { subject.commit(txn_id).await }),
                )
                .await;
            }
        }
    }

    async fn finalize(&self, txn_id: &TxnId) {
        debug!("finalize chain subject");

        match self {
            Self::Collection(subject) => subject.finalize(txn_id).await,
            Self::Dynamic(subject) => subject.finalize(txn_id).await,
            Self::Map(map) => {
                join_all(map.iter().map(|(_, subject)| async move {
                    subject.finalize(txn_id).await;
                }))
                .await;
            }
            Self::Tuple(tuple) => {
                join_all(tuple.iter().map(|subject| async move {
                    subject.finalize(txn_id).await;
                }))
                .await;
            }
        }
    }
}

#[async_trait]
impl de::FromStream for Subject {
    type Context = Txn;

    async fn from_stream<D: de::Decoder>(txn: Txn, decoder: &mut D) -> Result<Self, D::Error> {
        let state = State::from_stream(txn, decoder).await?;
        Self::from_state(state).map_err(de::Error::custom)
    }
}

impl From<SubjectCollection> for Subject {
    fn from(subject: SubjectCollection) -> Self {
        Self::Collection(subject)
    }
}

#[async_trait]
impl<'en> IntoView<'en, fs::Dir> for Subject {
    type Txn = Txn;
    type View = StateView<'en>;

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        match self {
            Self::Collection(subject) => subject.into_view(txn).await,
            Self::Dynamic(subject) => subject.into_view(txn).await,
            Self::Map(map) => {
                let views = map.into_iter().map(|(name, subject)| {
                    let txn = txn.clone();
                    async move { subject.into_view(txn).map_ok(|view| (name, view)).await }
                });

                let views = try_join_all(views).await?;
                Ok(StateView::Map(views.into_iter().collect()))
            }
            Self::Tuple(tuple) => {
                let views = tuple
                    .into_iter()
                    .map(|subject| subject.into_view(txn.clone()));

                try_join_all(views).map_ok(StateView::Tuple).await
            }
        }
    }
}

impl fmt::Display for Subject {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Collection(subject) => fmt::Display::fmt(subject, f),
            Self::Dynamic(_) => f.write_str("a dynamic Chain subject"),
            Self::Map(map) => fmt::Display::fmt(map, f),
            Self::Tuple(tuple) => fmt::Display::fmt(tuple, f),
        }
    }
}
