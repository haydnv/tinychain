use std::fmt;
use std::marker::PhantomData;

use async_trait::async_trait;
use destream::en;
use futures::future::TryFutureExt;
use log::debug;
use safecast::*;

use tc_collection::btree::Node as BTreeNode;
use tc_collection::tensor::{DenseCacheFile, Node as TensorNode};
use tc_collection::{btree, Collection, CollectionBase, CollectionView, Schema};
use tc_error::*;
use tc_scalar::{OpRef, Scalar, TCRef};
use tc_transact::fs;
use tc_transact::public::StateInstance;
use tc_transact::{AsyncHash, IntoView, Transact, Transaction, TxnId};
use tc_value::Value;
use tcgeneric::{Id, Instance, NativeClass, ThreadSafe};

pub enum StoreEntry<Txn, FE> {
    Collection(Collection<Txn, FE>),
    Scalar(Scalar),
}

impl<Txn, FE> Clone for StoreEntry<Txn, FE>
where
    Collection<Txn, FE>: Clone,
{
    fn clone(&self) -> Self {
        match self {
            Self::Collection(collection) => Self::Collection(collection.clone()),
            Self::Scalar(scalar) => Self::Scalar(scalar.clone()),
        }
    }
}

impl<Txn, FE> StoreEntry<Txn, FE> {
    pub fn try_from_state<State>(state: State) -> TCResult<Self>
    where
        State: StateInstance<Txn = Txn, FE = FE>,
        Txn: Transaction<FE>,
        FE: DenseCacheFile + AsType<btree::Node> + AsType<TensorNode> + Clone,
        Collection<Txn, FE>: TryCastFrom<State>,
        Scalar: TryCastFrom<State>,
        BTreeNode: freqfs::FileLoad,
    {
        if Collection::<_, _>::can_cast_from(&state) {
            state
                .try_cast_into(|s| bad_request!("not a collection: {s:?}"))
                .map(Self::Collection)
        } else if Scalar::can_cast_from(&state) {
            state
                .try_cast_into(|s| bad_request!("not a scalar: {s:?}"))
                .map(Self::Scalar)
        } else {
            Err(bad_request!("invalid Chain value entry: {state:?}"))
        }
    }

    pub fn into_state<State>(self) -> State
    where
        State: StateInstance<Txn = Txn, FE = FE> + From<Collection<Txn, FE>> + From<Scalar>,
    {
        match self {
            Self::Collection(collection) => collection.into(),
            Self::Scalar(scalar) => scalar.into(),
        }
    }
}

#[async_trait]
impl<'a, Txn, FE> AsyncHash for &'a StoreEntry<Txn, FE>
where
    FE: DenseCacheFile + AsType<BTreeNode> + AsType<TensorNode> + Clone,
    Txn: Transaction<FE>,
    Collection<Txn, FE>: AsyncHash,
    Scalar: async_hash::Hash<async_hash::Sha256>,
{
    async fn hash(self, txn_id: TxnId) -> TCResult<async_hash::Output<async_hash::Sha256>> {
        match self {
            StoreEntry::Collection(collection) => collection.clone().hash(txn_id).await,
            StoreEntry::Scalar(scalar) => Ok(async_hash::Hash::<async_hash::Sha256>::hash(scalar)),
        }
    }
}

#[async_trait]
impl<'en, Txn, FE> IntoView<'en, FE> for StoreEntry<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<BTreeNode> + AsType<TensorNode> + Clone,
{
    type Txn = Txn;
    type View = StoreEntryView<'en>;

    async fn into_view(self, txn: Self::Txn) -> TCResult<Self::View> {
        match self {
            Self::Collection(collection) => {
                collection
                    .into_view(txn)
                    .map_ok(StoreEntryView::Collection)
                    .await
            }
            Self::Scalar(scalar) => Ok(StoreEntryView::Scalar(scalar)),
        }
    }
}

impl<Txn, FE> fmt::Debug for StoreEntry<Txn, FE> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Collection(collection) => collection.fmt(f),
            Self::Scalar(scalar) => scalar.fmt(f),
        }
    }
}

pub enum StoreEntryView<'en> {
    Collection(CollectionView<'en>),
    Scalar(Scalar),
}

impl<'en> en::IntoStream<'en> for StoreEntryView<'en> {
    fn into_stream<E: en::Encoder<'en>>(self, encoder: E) -> Result<E::Ok, E::Error> {
        match self {
            Self::Collection(collection) => collection.into_stream(encoder),
            Self::Scalar(scalar) => scalar.into_stream(encoder),
        }
    }
}

pub struct Store<Txn, FE> {
    dir: fs::Dir<FE>,
    txn: PhantomData<Txn>,
}

impl<Txn, FE> Clone for Store<Txn, FE> {
    fn clone(&self) -> Self {
        Self {
            dir: self.dir.clone(),
            txn: self.txn.clone(),
        }
    }
}

impl<Txn, FE> Store<Txn, FE> {
    pub fn new(dir: fs::Dir<FE>) -> Self {
        Self {
            dir,
            txn: PhantomData,
        }
    }
}

impl<Txn, FE> Store<Txn, FE>
where
    Txn: Transaction<FE>,
    FE: DenseCacheFile + AsType<BTreeNode> + AsType<TensorNode> + Clone,
    BTreeNode: freqfs::FileLoad,
{
    pub async fn resolve(&self, txn_id: TxnId, scalar: Scalar) -> TCResult<StoreEntry<Txn, FE>> {
        debug!("History::resolve {:?}", scalar);

        type OpSubject = tc_scalar::Subject;

        if let Scalar::Ref(tc_ref) = scalar {
            if let TCRef::Op(OpRef::Get((OpSubject::Ref(hash, classpath), schema))) = *tc_ref {
                let hash = hash.into_id();
                let store = self.dir.get_dir(txn_id, &hash).await?;
                let schema = Value::try_cast_from(schema, |s| {
                    internal!("invalid schema for Collection: {s:?}")
                })
                .and_then(|schema| Schema::try_from((classpath, schema)))?;

                <CollectionBase<Txn, FE> as fs::Persist<FE>>::load(txn_id, schema, store)
                    .map_ok(Collection::from)
                    .map_ok(StoreEntry::Collection)
                    .await
            } else {
                Err(internal!(
                    "invalid subject for historical Chain state {:?}",
                    tc_ref
                ))
            }
        } else {
            Ok(StoreEntry::Scalar(scalar))
        }
    }

    pub async fn save_state(&self, txn: &Txn, state: StoreEntry<Txn, FE>) -> TCResult<Scalar> {
        debug!("chain data store saving state {:?}...", state);

        match state {
            StoreEntry::Collection(collection) => {
                let classpath = collection.class().path();
                let schema = collection.schema();

                let txn_id = *txn.id();
                let hash = collection.clone().hash(txn_id).map_ok(Id::from).await?;

                if !self.dir.contains(txn_id, &hash).await? {
                    let store = self.dir.create_dir(txn_id, hash.clone()).await?;
                    let _copy: CollectionBase<_, _> =
                        fs::CopyFrom::copy_from(txn, store, collection).await?;
                }

                Ok(OpRef::Get((
                    (hash.into(), classpath).into(),
                    Value::cast_from(schema).into(),
                ))
                .into())
            }
            StoreEntry::Scalar(scalar) => Ok(scalar),
        }
    }
}

#[async_trait]
impl<Txn, FE> Transact for Store<Txn, FE>
where
    FE: ThreadSafe + Clone + for<'a> fs::FileSave<'a>,
    Txn: Transaction<FE>,
{
    type Commit = ();

    async fn commit(&self, txn_id: TxnId) -> Self::Commit {
        debug!("commit chain data store at {}", txn_id);
        self.dir.commit(txn_id, true).await
    }

    async fn rollback(&self, txn_id: &TxnId) {
        self.dir.rollback(*txn_id, true).await
    }

    async fn finalize(&self, txn_id: &TxnId) {
        self.dir.finalize(*txn_id).await
    }
}
