use std::fmt;
use std::marker::PhantomData;

use log::debug;
use safecast::AsType;
use tc_chain::ChainBlock;
use tc_collection::{BTreeNode, DenseCacheFile, TensorNode};

use tc_error::*;
use tc_scalar::op::*;
use tc_scalar::Scalar;
use tc_transact::public::{
    DeleteHandler, GetHandler, Handler, PostHandler, PutHandler, Route, ToState,
};
use tc_transact::{fs, RPCClient, Transaction};
use tc_value::Value;
use tcgeneric::{Id, Instance, Map, PathSegment};

use crate::State;

struct GetMethod<'a, Txn, FE, T: Instance> {
    subject: &'a T,
    name: &'a Id,
    method: GetOp,
    phantom: PhantomData<(Txn, FE)>,
}

impl<'a, Txn, FE, T: Instance> GetMethod<'a, Txn, FE, T> {
    pub fn new(subject: &'a T, name: &'a Id, method: GetOp) -> Self {
        Self {
            subject,
            name,
            method,
            phantom: PhantomData,
        }
    }
}

impl<'a, Txn, FE, T> GetMethod<'a, Txn, FE, T>
where
    Txn: Transaction<FE> + RPCClient<State<Txn, FE>>,
    FE: DenseCacheFile
        + AsType<BTreeNode>
        + AsType<ChainBlock>
        + AsType<TensorNode>
        + for<'b> fs::FileSave<'b>
        + Clone,
    T: ToState<State<Txn, FE>> + Instance + Route<State<Txn, FE>> + fmt::Debug + 'a,
{
    async fn call(self, txn: &Txn, key: Value) -> TCResult<State<Txn, FE>> {
        let (key_name, op_def) = self.method;

        let mut context = Map::new();
        context.insert(key_name, key.into());

        match call_method(txn, self.subject, context, op_def).await {
            Ok(state) => Ok(state),
            Err(cause) => {
                Err(cause.consume(format!("in call to GET {:?} /{}", self.subject, self.name)))
            }
        }
    }
}

impl<'a, Txn, FE, T> Handler<'a, State<Txn, FE>> for GetMethod<'a, Txn, FE, T>
where
    Txn: Transaction<FE> + RPCClient<State<Txn, FE>>,
    FE: DenseCacheFile
        + AsType<BTreeNode>
        + AsType<ChainBlock>
        + AsType<TensorNode>
        + for<'b> fs::FileSave<'b>
        + Clone,
    T: ToState<State<Txn, FE>> + Instance + Route<State<Txn, FE>> + fmt::Debug + 'a,
{
    fn get<'b>(self: Box<Self>) -> Option<GetHandler<'a, 'b, Txn, State<Txn, FE>>>
    where
        'b: 'a,
    {
        Some(Box::new(move |txn, key| Box::pin(self.call(txn, key))))
    }
}

struct PutMethod<'a, Txn, FE, T: Instance> {
    subject: &'a T,
    name: &'a Id,
    method: PutOp,
    phantom: PhantomData<(Txn, FE)>,
}

impl<'a, Txn, FE, T: Instance> PutMethod<'a, Txn, FE, T> {
    pub fn new(subject: &'a T, name: &'a Id, method: PutOp) -> Self {
        Self {
            subject,
            name,
            method,
            phantom: PhantomData,
        }
    }
}

impl<'a, Txn, FE, T> PutMethod<'a, Txn, FE, T>
where
    Txn: Transaction<FE> + RPCClient<State<Txn, FE>>,
    FE: DenseCacheFile
        + AsType<BTreeNode>
        + AsType<ChainBlock>
        + AsType<TensorNode>
        + for<'b> fs::FileSave<'b>
        + Clone,
    T: ToState<State<Txn, FE>> + Instance + Route<State<Txn, FE>> + fmt::Debug + 'a,
{
    async fn call(self, txn: &Txn, key: Value, value: State<Txn, FE>) -> TCResult<()> {
        let (key_name, value_name, op_def) = self.method;

        let mut context = Map::new();
        context.insert(key_name, key.into());
        context.insert(value_name, value);

        match call_method(txn, self.subject, context, op_def).await {
            Ok(_) => Ok(()),
            Err(cause) => {
                Err(cause.consume(format!("in call to PUT {:?} /{}", self.subject, self.name)))
            }
        }
    }
}

impl<'a, Txn, FE, T> Handler<'a, State<Txn, FE>> for PutMethod<'a, Txn, FE, T>
where
    Txn: Transaction<FE> + RPCClient<State<Txn, FE>>,
    FE: DenseCacheFile
        + AsType<BTreeNode>
        + AsType<ChainBlock>
        + AsType<TensorNode>
        + for<'b> fs::FileSave<'b>
        + Clone,
    T: ToState<State<Txn, FE>> + Instance + Route<State<Txn, FE>> + fmt::Debug + 'a,
{
    fn put<'b>(self: Box<Self>) -> Option<PutHandler<'a, 'b, Txn, State<Txn, FE>>>
    where
        'b: 'a,
    {
        Some(Box::new(move |txn, key, value| {
            Box::pin(self.call(txn, key, value))
        }))
    }
}

struct PostMethod<'a, Txn, FE, T: Instance> {
    subject: &'a T,
    name: &'a Id,
    method: PostOp,
    phantom: PhantomData<(Txn, FE)>,
}

impl<'a, Txn, FE, T: Instance> PostMethod<'a, Txn, FE, T> {
    pub fn new(subject: &'a T, name: &'a Id, method: PostOp) -> Self {
        Self {
            subject,
            name,
            method,
            phantom: PhantomData,
        }
    }
}

impl<'a, Txn, FE, T> PostMethod<'a, Txn, FE, T>
where
    Txn: Transaction<FE> + RPCClient<State<Txn, FE>>,
    FE: DenseCacheFile
        + AsType<BTreeNode>
        + AsType<ChainBlock>
        + AsType<TensorNode>
        + for<'b> fs::FileSave<'b>
        + Clone,
    T: ToState<State<Txn, FE>> + Instance + Route<State<Txn, FE>> + fmt::Debug + 'a,
{
    async fn call(self, txn: &Txn, params: Map<State<Txn, FE>>) -> TCResult<State<Txn, FE>> {
        match call_method(txn, self.subject, params, self.method).await {
            Ok(state) => Ok(state),
            Err(cause) => {
                Err(cause.consume(format!("in call to POST {:?} /{}", self.subject, self.name)))
            }
        }
    }
}

impl<'a, Txn, FE, T> Handler<'a, State<Txn, FE>> for PostMethod<'a, Txn, FE, T>
where
    Txn: Transaction<FE> + RPCClient<State<Txn, FE>>,
    FE: DenseCacheFile
        + AsType<BTreeNode>
        + AsType<ChainBlock>
        + AsType<TensorNode>
        + for<'b> fs::FileSave<'b>
        + Clone,
    T: ToState<State<Txn, FE>> + Instance + Route<State<Txn, FE>> + fmt::Debug + 'a,
{
    fn post<'b>(self: Box<Self>) -> Option<PostHandler<'a, 'b, Txn, State<Txn, FE>>>
    where
        'b: 'a,
    {
        Some(Box::new(move |txn, params| {
            Box::pin(self.call(txn, params))
        }))
    }
}

struct DeleteMethod<'a, Txn, FE, T: Instance> {
    subject: &'a T,
    name: &'a Id,
    method: DeleteOp,
    phantom: PhantomData<(Txn, FE)>,
}

impl<'a, Txn, FE, T: Instance> DeleteMethod<'a, Txn, FE, T> {
    pub fn new(subject: &'a T, name: &'a Id, method: DeleteOp) -> Self {
        Self {
            subject,
            name,
            method,
            phantom: PhantomData,
        }
    }
}

impl<'a, Txn, FE, T> DeleteMethod<'a, Txn, FE, T>
where
    Txn: Transaction<FE> + RPCClient<State<Txn, FE>>,
    FE: DenseCacheFile
        + AsType<BTreeNode>
        + AsType<ChainBlock>
        + AsType<TensorNode>
        + for<'b> fs::FileSave<'b>
        + Clone,
    T: ToState<State<Txn, FE>> + Instance + Route<State<Txn, FE>> + fmt::Debug + 'a,
{
    async fn call(self, txn: &Txn, key: Value) -> TCResult<()> {
        let (key_name, op_def) = self.method;

        let mut context = Map::new();
        context.insert(key_name, key.into());

        match call_method(txn, self.subject, context, op_def).await {
            Ok(_) => Ok(()),
            Err(cause) => Err(cause.consume(format!(
                "in call to DELETE {:?} /{}",
                self.subject, self.name
            ))),
        }
    }
}

impl<'a, Txn, FE, T> Handler<'a, State<Txn, FE>> for DeleteMethod<'a, Txn, FE, T>
where
    Txn: Transaction<FE> + RPCClient<State<Txn, FE>>,
    FE: DenseCacheFile
        + AsType<BTreeNode>
        + AsType<ChainBlock>
        + AsType<TensorNode>
        + for<'b> fs::FileSave<'b>
        + Clone,
    T: ToState<State<Txn, FE>> + Instance + Route<State<Txn, FE>> + fmt::Debug + 'a,
{
    fn delete<'b>(self: Box<Self>) -> Option<DeleteHandler<'a, 'b, Txn>>
    where
        'b: 'a,
    {
        Some(Box::new(move |txn, key| Box::pin(self.call(txn, key))))
    }
}

#[inline]
pub fn route_attr<'a, Txn, FE, T>(
    subject: &'a T,
    name: &'a Id,
    attr: &'a Scalar,
    path: &'a [PathSegment],
) -> Option<Box<dyn Handler<'a, State<Txn, FE>> + 'a>>
where
    Txn: Transaction<FE> + RPCClient<State<Txn, FE>>,
    FE: DenseCacheFile
        + AsType<BTreeNode>
        + AsType<ChainBlock>
        + AsType<TensorNode>
        + for<'b> fs::FileSave<'b>
        + Clone,
    T: ToState<State<Txn, FE>> + Instance + Route<State<Txn, FE>> + fmt::Debug + 'a,
{
    match attr {
        Scalar::Op(OpDef::Get(get_op)) if path.is_empty() => {
            debug!("call GET method with subject {:?}", subject);

            Some(Box::new(GetMethod::new(subject, name, get_op.clone())))
        }
        Scalar::Op(OpDef::Put(put_op)) if path.is_empty() => {
            debug!("call PUT method with subject {:?}", subject);

            Some(Box::new(PutMethod::new(subject, name, put_op.clone())))
        }
        Scalar::Op(OpDef::Post(post_op)) if path.is_empty() => {
            debug!("call POST method with subject {:?}", subject);

            Some(Box::new(PostMethod::new(subject, name, post_op.clone())))
        }
        Scalar::Op(OpDef::Delete(delete_op)) if path.is_empty() => {
            debug!("call DELETE method with subject {:?}", subject);

            Some(Box::new(DeleteMethod::new(
                subject,
                name,
                delete_op.clone(),
            )))
        }
        other => other.route(path),
    }
}

async fn call_method<Txn, FE, T>(
    txn: &Txn,
    subject: &T,
    context: Map<State<Txn, FE>>,
    form: Vec<(Id, Scalar)>,
) -> TCResult<State<Txn, FE>>
where
    Txn: Transaction<FE> + RPCClient<State<Txn, FE>>,
    FE: DenseCacheFile
        + AsType<BTreeNode>
        + AsType<ChainBlock>
        + AsType<TensorNode>
        + for<'a> fs::FileSave<'a>
        + Clone,
    T: ToState<State<Txn, FE>> + Route<State<Txn, FE>> + Instance + fmt::Debug,
{
    debug!(
        "call method with form {}",
        form.iter()
            .map(|(id, s)| format!("{}: {:?}", id, s))
            .collect::<Vec<String>>()
            .join("\n")
    );

    let capture = if let Some((capture, _)) = form.last() {
        capture.clone()
    } else {
        return Ok(State::default());
    };

    Executor::with_context(txn, Some(subject), context.into(), form)
        .capture(capture)
        .await
}
