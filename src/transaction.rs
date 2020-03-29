pub struct TransactionId {
    timestamp: u64, // nanoseconds since Unix epoch
    nonce: u16,
}

pub struct Transaction {
    id: TransactionId,
}
