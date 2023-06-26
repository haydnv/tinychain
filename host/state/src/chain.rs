use super::State;

pub type Chain<T> = tc_chain::Chain<State, T>;
pub type BlockChain<T> = tc_chain::BlockChain<State, T>;
pub type SyncChain<T> = tc_chain::SyncChain<State, T>;
