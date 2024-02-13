use tc_value::Link;
use tcgeneric::NetworkTime;

pub struct Gateway {
    authorized: Vec<Link>,
}

impl Gateway {
    pub fn new() -> Self {
        Self { authorized: vec![] }
    }

    pub fn now(&self) -> NetworkTime {
        NetworkTime::now()
    }
}
