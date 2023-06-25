use crate::gateway::Gateway;
use crate::state::State;

pub type Hypothetical = tc_fs::hypothetical::Hypothetical<State>;
pub type Txn = tc_fs::Txn<Gateway>;
