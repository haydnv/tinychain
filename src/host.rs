use crate::context::TCContext;
use crate::drive::Drive;

pub struct HostContext {
    workspace: Drive,
}

impl HostContext {
    pub fn new(workspace: Drive) -> HostContext {
        HostContext { workspace }
    }
}

impl TCContext for HostContext {}
