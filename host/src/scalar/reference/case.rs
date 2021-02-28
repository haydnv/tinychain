use std::collections::HashSet;
use std::convert::TryFrom;

use async_trait::async_trait;

use tc_error::TCResult;
use tcgeneric::{Id, Instance, Tuple};

use super::Refer;
use crate::route::Public;
use crate::scalar::{Scalar, Scope, Value};
use crate::state::State;
use crate::txn::Txn;

pub struct Case {
    cond: Scalar,
    switch: Tuple<Scalar>,
    case: Tuple<Scalar>,
}

#[async_trait]
impl Refer for Case {
    fn requires(&self, deps: &mut HashSet<Id>) {
        self.cond.requires(deps);

        for switch in self.switch.iter() {
            switch.requires(deps);
        }
    }

    async fn resolve<'a, T: Public + Instance>(
        mut self,
        context: &'a Scope<'a, T>,
        txn: &'a Txn,
    ) -> TCResult<State> {
        assert_eq!(self.switch.len() + 1, self.case.len());

        let cond = self.cond.resolve(context, txn).await?;
        let cond = Value::try_from(cond)?;
        for (i, switch) in self.switch.into_iter().enumerate() {
            let switch = switch.resolve(context, txn).await?;
            let switch = Value::try_from(switch)?;
            if cond == switch {
                return self.case.remove(i).resolve(context, txn).await;
            }
        }

        let case = self.case.pop().unwrap();
        case.resolve(context, txn).await
    }
}
