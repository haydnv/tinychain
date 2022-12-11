//! [`Link`] and its components

use std::cmp::Ordering;
use std::convert::TryFrom;
use std::fmt;
use std::iter;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::str::FromStr;

use async_hash::Hash;
use async_trait::async_trait;
use destream::{de, en};
use number_general::Number;
use safecast::{CastFrom, TryCastFrom};
use serde::de::{Deserialize, Deserializer};
use serde::ser::{Serialize, Serializer};
use sha2::digest::{Digest, Output};

use tc_error::*;
use tcgeneric::{Id, PathLabel, PathSegment, TCPathBuf};

use super::Value;

/// The address portion of a [`Link`] (an IP address)
#[derive(Debug, Hash, Eq, PartialEq)]
pub enum LinkAddress {
    IPv4(Ipv4Addr),
    IPv6(Ipv6Addr),
}

impl Clone for LinkAddress {
    fn clone(&self) -> Self {
        use LinkAddress::*;
        match self {
            IPv4(addr) => IPv4(*addr),
            IPv6(addr) => IPv6(*addr),
        }
    }
}

impl fmt::Display for LinkAddress {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use LinkAddress::*;

        match self {
            IPv4(addr) => write!(f, "{}", addr),
            IPv6(addr) => write!(f, "{}", addr),
        }
    }
}

impl From<Ipv4Addr> for LinkAddress {
    fn from(addr: Ipv4Addr) -> LinkAddress {
        LinkAddress::IPv4(addr)
    }
}

impl From<Ipv6Addr> for LinkAddress {
    fn from(addr: Ipv6Addr) -> LinkAddress {
        LinkAddress::IPv6(addr)
    }
}

impl From<IpAddr> for LinkAddress {
    fn from(addr: IpAddr) -> LinkAddress {
        use IpAddr::*;
        use LinkAddress::*;

        match addr {
            V4(addr) => IPv4(addr),
            V6(addr) => IPv6(addr),
        }
    }
}

impl PartialEq<Ipv4Addr> for LinkAddress {
    fn eq(&self, other: &Ipv4Addr) -> bool {
        use LinkAddress::*;

        match self {
            IPv4(addr) => addr == other,
            _ => false,
        }
    }
}

impl PartialEq<Ipv6Addr> for LinkAddress {
    fn eq(&self, other: &Ipv6Addr) -> bool {
        use LinkAddress::*;

        match self {
            IPv6(addr) => addr == other,
            _ => false,
        }
    }
}

impl PartialEq<IpAddr> for LinkAddress {
    fn eq(&self, other: &IpAddr) -> bool {
        use IpAddr::*;

        match other {
            V4(addr) => self == addr,
            V6(addr) => self == addr,
        }
    }
}

/// The protocol portion of a [`Link`] (e.g. "http")
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub enum LinkProtocol {
    HTTP,
}

impl Default for LinkProtocol {
    fn default() -> LinkProtocol {
        LinkProtocol::HTTP
    }
}

impl fmt::Display for LinkProtocol {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                LinkProtocol::HTTP => "http",
            }
        )
    }
}

/// The host portion of a [`Link`] (e.g. "http://127.0.0.1:8702")
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct LinkHost {
    protocol: LinkProtocol,
    address: LinkAddress,
    port: Option<u16>,
}

impl LinkHost {
    pub fn new(protocol: LinkProtocol, address: LinkAddress, port: Option<u16>) -> Self {
        Self {
            protocol,
            address,
            port,
        }
    }

    pub fn address(&'_ self) -> &'_ LinkAddress {
        &self.address
    }

    pub fn authority(&self) -> String {
        if let Some(port) = self.port {
            format!("{}:{}", self.address, port)
        } else {
            self.address.to_string()
        }
    }

    pub fn port(&'_ self) -> &'_ Option<u16> {
        &self.port
    }

    pub fn protocol(&'_ self) -> &'_ LinkProtocol {
        &self.protocol
    }
}

impl FromStr for LinkHost {
    type Err = TCError;

    fn from_str(s: &str) -> TCResult<LinkHost> {
        if !s.starts_with("http://") {
            return Err(TCError::bad_request("Unable to parse Link protocol", s));
        }

        let protocol = LinkProtocol::HTTP;

        let s = &s[7..];

        let (address, port): (LinkAddress, Option<u16>) = if s.contains("::") {
            let mut segments: Vec<&str> = s.split("::").collect();
            let port: Option<u16> = if segments.last().unwrap().contains(':') {
                let last_segment: Vec<&str> = segments.pop().unwrap().split(':').collect();
                if last_segment.len() == 2 {
                    segments.push(last_segment[0]);
                    Some(
                        last_segment[1]
                            .parse()
                            .map_err(|e| TCError::bad_request("Unable to parse port number", e))?,
                    )
                } else {
                    return Err(TCError::bad_request("Unable to parse IPv6 address", s));
                }
            } else {
                None
            };

            let address: Ipv6Addr = segments
                .join("::")
                .parse()
                .map_err(|e| TCError::bad_request("Unable to parse IPv6 address", e))?;

            (address.into(), port)
        } else {
            let (address, port) = if s.contains(':') {
                let segments: Vec<&str> = s.split(':').collect();
                if segments.len() == 2 {
                    let port: u16 = segments[1]
                        .parse()
                        .map_err(|e| TCError::bad_request("Unable to parse port number", e))?;

                    (segments[0], Some(port))
                } else {
                    return Err(TCError::bad_request("Unable to parse network address", s));
                }
            } else {
                (s, None)
            };

            let address: Ipv4Addr = address
                .parse()
                .map_err(|e| TCError::bad_request("Unable to parse IPv4 address", e))?;

            (address.into(), port)
        };

        Ok(LinkHost {
            protocol,
            address,
            port,
        })
    }
}

impl<A: Into<LinkAddress>> From<(A, u16)> for LinkHost {
    fn from(addr: (A, u16)) -> LinkHost {
        LinkHost {
            protocol: LinkProtocol::default(),
            address: addr.0.into(),
            port: Some(addr.1),
        }
    }
}

impl<A: Into<LinkAddress>> From<(LinkProtocol, A, Option<u16>)> for LinkHost {
    fn from(addr: (LinkProtocol, A, Option<u16>)) -> LinkHost {
        LinkHost {
            protocol: addr.0,
            address: addr.1.into(),
            port: addr.2,
        }
    }
}

impl TryFrom<Link> for LinkHost {
    type Error = TCError;

    fn try_from(link: Link) -> TCResult<LinkHost> {
        if link.path == TCPathBuf::default() {
            if let Some(host) = link.host() {
                Ok(host.clone())
            } else {
                Err(TCError::bad_request("This Link has no LinkHost", link))
            }
        } else {
            Err(TCError::bad_request(
                "Cannot convert to LinkHost without losing path information",
                link,
            ))
        }
    }
}

impl<'de> Deserialize<'de> for LinkHost {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

impl Serialize for LinkHost {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.to_string().serialize(serializer)
    }
}

impl fmt::Display for LinkHost {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(port) = self.port() {
            write!(f, "{}://{}:{}", self.protocol, self.address, port)
        } else {
            write!(f, "{}://{}", self.protocol, self.address)
        }
    }
}

/// A link to a network resource.
#[derive(Clone, Debug, Default, Hash, Eq, PartialEq)]
pub struct Link {
    host: Option<LinkHost>,
    path: TCPathBuf,
}

impl Link {
    /// Create a new [`Link`] with the given [`LinkHost`] and [`TCPathBuf`].
    pub fn new(host: LinkHost, path: TCPathBuf) -> Self {
        Self {
            host: Some(host),
            path,
        }
    }

    /// Consume this `Link` and return its path.
    pub fn into_path(self) -> TCPathBuf {
        self.path
    }

    /// Borrow this `Link`'s [`LinkHost`], if it has one.
    pub fn host(&'_ self) -> &'_ Option<LinkHost> {
        &self.host
    }

    /// Borrow this `Link`'s path.
    pub fn path(&'_ self) -> &'_ TCPathBuf {
        &self.path
    }

    /// Append the given [`PathSegment`] to this `Link` and return it.
    pub fn append<S: Into<PathSegment>>(mut self, segment: S) -> Self {
        self.path = self.path.append(segment);
        self
    }
}

impl Extend<PathSegment> for Link {
    fn extend<T: IntoIterator<Item = PathSegment>>(&mut self, iter: T) {
        self.path.extend(iter)
    }
}

impl Ord for Link {
    fn cmp(&self, other: &Self) -> Ordering {
        self.to_string().cmp(&other.to_string())
    }
}

impl PartialEq<[PathSegment]> for Link {
    fn eq(&self, other: &[PathSegment]) -> bool {
        if self.host.is_some() {
            return false;
        }

        &self.path == other
    }
}

impl PartialOrd for Link {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.to_string().partial_cmp(&other.to_string())
    }
}

impl<'a, D: Digest> Hash<D> for &'a Link {
    fn hash(self) -> Output<D> {
        Hash::<D>::hash(self.to_string().as_str())
    }
}

impl From<LinkHost> for Link {
    fn from(host: LinkHost) -> Link {
        Link {
            host: Some(host),
            path: TCPathBuf::default(),
        }
    }
}

impl From<PathLabel> for Link {
    fn from(path: PathLabel) -> Self {
        TCPathBuf::from(path).into()
    }
}

impl<A: Into<LinkAddress>> From<(A, u16)> for Link {
    fn from(addr: (A, u16)) -> Link {
        Link {
            host: Some(addr.into()),
            path: TCPathBuf::default(),
        }
    }
}

impl From<TCPathBuf> for Link {
    fn from(path: TCPathBuf) -> Link {
        Link { host: None, path }
    }
}

impl From<(LinkHost, TCPathBuf)> for Link {
    fn from(tuple: (LinkHost, TCPathBuf)) -> Link {
        let (host, path) = tuple;
        Link {
            host: Some(host),
            path,
        }
    }
}

impl TryCastFrom<Value> for Link {
    fn can_cast_from(value: &Value) -> bool {
        match value {
            Value::Bytes(_) => false,
            Value::Email(email) => Self::from_str(email.get_domain()).is_ok(),
            Value::Id(_) => true,
            Value::Link(_) => true,
            Value::None => true,
            Value::Number(n) => match n {
                Number::UInt(_) => true,
                _ => false,
            },
            Value::String(s) => Self::from_str(s).is_ok(),
            Value::Tuple(t) => t.iter().all(|v| Id::can_cast_from(v)),
            Value::Version(_) => true,
        }
    }

    fn opt_cast_from(value: Value) -> Option<Self> {
        match value {
            Value::Bytes(_) => None,
            Value::Email(email) => Self::from_str(email.get_domain()).ok(),
            Value::Id(id) => Some(TCPathBuf::from(id).into()),
            Value::Link(l) => Some(l),
            Value::None => Some(TCPathBuf::default().into()),
            Value::Number(n) => match n {
                Number::UInt(u) => Some(TCPathBuf::from(vec![Id::from(u64::cast_from(u))]).into()),
                _ => None,
            },
            Value::String(s) => Self::from_str(&s).ok(),
            Value::Tuple(t) => {
                let mut path = Vec::with_capacity(t.len());
                for id in t.into_iter() {
                    if let Some(id) = Id::opt_cast_from(id) {
                        path.push(id);
                    } else {
                        return None;
                    }
                }

                Some(TCPathBuf::from(path).into())
            }
            Value::Version(version) => Some(TCPathBuf::from(Id::from(version)).into()),
        }
    }
}

impl FromStr for Link {
    type Err = TCError;

    fn from_str(s: &str) -> TCResult<Link> {
        if s.starts_with('/') {
            return Ok(Link {
                host: None,
                path: s.parse()?,
            });
        } else if !s.starts_with("http://") {
            return Err(TCError::bad_request("Unable to parse Link protocol", s));
        }

        let s = if s.ends_with('/') {
            &s[..s.len() - 1]
        } else {
            s
        };

        let segments: Vec<&str> = s.split('/').collect();
        if segments.is_empty() {
            return Err(TCError::bad_request("Unable to parse Link", s));
        }

        let host: LinkHost = segments[..3].join("/").parse()?;

        let segments = &segments[3..];

        let segments = segments
            .iter()
            .map(|s| s.parse())
            .map(|r| r.map_err(TCError::unsupported))
            .collect::<TCResult<Vec<PathSegment>>>()?;

        Ok(Link {
            host: Some(host),
            path: iter::FromIterator::from_iter(segments),
        })
    }
}

impl<'de> Deserialize<'de> for Link {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let link = String::deserialize(deserializer)?;
        Self::from_str(&link).map_err(serde::de::Error::custom)
    }
}

impl Serialize for Link {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.to_string().serialize(serializer)
    }
}

struct LinkVisitor;

#[async_trait]
impl de::Visitor for LinkVisitor {
    type Value = Link;

    fn expecting() -> &'static str {
        "a Link"
    }

    fn visit_string<E: de::Error>(self, s: String) -> Result<Self::Value, E> {
        s.parse().map_err(de::Error::custom)
    }

    async fn visit_map<A: de::MapAccess>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let s = map
            .next_key(())
            .await?
            .ok_or_else(|| de::Error::invalid_length(0, Self::expecting()))?;

        let _ = map.next_value::<[Value; 0]>(()).await?;
        self.visit_string(s)
    }
}

#[async_trait]
impl de::FromStream for Link {
    type Context = ();

    async fn from_stream<D: de::Decoder>(_: (), decoder: &mut D) -> Result<Link, D::Error> {
        decoder.decode_any(LinkVisitor).await
    }
}

impl<'en> en::ToStream<'en> for Link {
    fn to_stream<E: en::Encoder<'en>>(&'en self, e: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream(self.to_string(), e)
    }
}

impl<'en> en::IntoStream<'en> for Link {
    fn into_stream<E: en::Encoder<'en>>(self, e: E) -> Result<E::Ok, E::Error> {
        en::IntoStream::into_stream(self.to_string(), e)
    }
}

impl fmt::Display for Link {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(host) = &self.host {
            write!(f, "{}", host)?;
        }

        write!(f, "{}", self.path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eq() {
        let a = LinkHost {
            protocol: LinkProtocol::HTTP,
            address: Ipv4Addr::new(1, 2, 3, 4).into(),
            port: Some(123),
        };

        let b = LinkHost {
            protocol: LinkProtocol::HTTP,
            address: Ipv4Addr::new(1, 2, 3, 4).into(),
            port: Some(234),
        };

        assert_ne!(a, b);
    }
}
