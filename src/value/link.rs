use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use std::fmt;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::path::PathBuf;
use std::str::FromStr;

use addr::DomainName;
use serde::de;
use serde::ser::{SerializeMap, Serializer};

use crate::error;
use crate::value::string::Label;
use crate::value::{TCResult, Value, ValueId};

pub type PathSegment = ValueId;

#[derive(Debug, Hash, Eq, PartialEq)]
pub enum LinkAddress {
    DomainName(DomainName),
    IPv4(Ipv4Addr),
    IPv6(Ipv6Addr),
}

impl Clone for LinkAddress {
    fn clone(&self) -> Self {
        use LinkAddress::*;
        match self {
            DomainName(n) => DomainName(n.to_string().parse().unwrap()),
            other => other.clone(),
        }
    }
}

impl fmt::Display for LinkAddress {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use LinkAddress::*;

        match self {
            DomainName(addr) => write!(f, "{}", addr),
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

impl From<DomainName> for LinkAddress {
    fn from(addr: DomainName) -> LinkAddress {
        LinkAddress::DomainName(addr)
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

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct LinkHost {
    protocol: LinkProtocol,
    address: LinkAddress,
    port: Option<u16>,
}

impl LinkHost {
    pub fn address(&'_ self) -> &'_ LinkAddress {
        &self.address
    }

    pub fn port(&'_ self) -> &'_ Option<u16> {
        &self.port
    }

    pub fn protocol(&'_ self) -> &'_ LinkProtocol {
        &self.protocol
    }
}

fn is_ipv4(s: &str) -> bool {
    let segments: Vec<&str> = s.split('.').collect();
    segments.len() == 4 && !segments.iter().any(|s| s.parse::<u16>().is_err())
}

impl FromStr for LinkHost {
    type Err = error::TCError;

    fn from_str(s: &str) -> TCResult<LinkHost> {
        if !s.starts_with("http://") {
            return Err(error::bad_request("Unable to parse Link protocol", s));
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
                            .map_err(|e| error::bad_request("Unable to parse port number", e))?,
                    )
                } else {
                    return Err(error::bad_request("Unable to parse IPv6 address", s));
                }
            } else {
                None
            };

            let address: Ipv6Addr = segments
                .join("::")
                .parse()
                .map_err(|e| error::bad_request("Unable to parse IPv6 address", e))?;
            (address.into(), port)
        } else {
            let (address, port) = if s.contains(':') {
                let segments: Vec<&str> = s.split(':').collect();
                if segments.len() == 2 {
                    let port: u16 = segments[1]
                        .parse()
                        .map_err(|e| error::bad_request("Unable to parse port number", e))?;
                    (segments[0], Some(port))
                } else {
                    return Err(error::bad_request("Unable to parse network address", s));
                }
            } else {
                (s, None)
            };

            let address: LinkAddress = if is_ipv4(address) {
                let address: Ipv4Addr = address
                    .parse()
                    .map_err(|e| error::bad_request("Unable to parse IPv4 address", e))?;
                address.into()
            } else {
                let address: DomainName = address
                    .parse()
                    .map_err(|e| error::bad_request("Unable to parse domain name", e))?;
                address.into()
            };

            (address, port)
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
    type Error = error::TCError;

    fn try_from(link: Link) -> TCResult<LinkHost> {
        if link.path == TCPath::default() {
            if let Some(host) = link.host() {
                Ok(host.clone())
            } else {
                Err(error::bad_request("This Link has no LinkHost", link))
            }
        } else {
            Err(error::bad_request(
                "Cannot convert to LinkHost without losing path information",
                link,
            ))
        }
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

#[derive(Clone, Debug, Default, Hash, Eq, PartialEq)]
pub struct Link {
    host: Option<LinkHost>,
    path: TCPath,
}

impl Link {
    pub fn into_path(self) -> TCPath {
        self.path
    }

    pub fn join(self, other: TCPath) -> Link {
        Link {
            host: self.host,
            path: self.path.join(other),
        }
    }

    pub fn host(&'_ self) -> &'_ Option<LinkHost> {
        &self.host
    }

    pub fn path(&'_ self) -> &'_ TCPath {
        &self.path
    }
}

impl From<LinkHost> for Link {
    fn from(host: LinkHost) -> Link {
        Link {
            host: Some(host),
            path: TCPath::default(),
        }
    }
}

impl<A: Into<LinkAddress>> From<(A, u16)> for Link {
    fn from(addr: (A, u16)) -> Link {
        Link {
            host: Some(addr.into()),
            path: TCPath::default(),
        }
    }
}

impl From<TCPath> for Link {
    fn from(path: TCPath) -> Link {
        Link { host: None, path }
    }
}

impl From<(LinkHost, TCPath)> for Link {
    fn from(tuple: (LinkHost, TCPath)) -> Link {
        let (host, path) = tuple;
        Link {
            host: Some(host),
            path,
        }
    }
}

impl FromStr for Link {
    type Err = error::TCError;

    fn from_str(s: &str) -> TCResult<Link> {
        if s.starts_with('/') {
            return Ok(Link {
                host: None,
                path: s.parse()?,
            });
        } else if !s.starts_with("http://") {
            return Err(error::bad_request("Unable to parse Link protocol", s));
        } else if s.ends_with('/') {
            return Err(error::bad_request(
                "Link is not allowed to end with a '/'",
                s,
            ));
        }

        let segments: Vec<&str> = s.split('/').collect();
        if segments.is_empty() {
            return Err(error::bad_request("Unable to parse Link", s));
        }

        let host: LinkHost = segments[..3].join("/").parse()?;
        let segments = &segments[3..];

        let path: TCPath = if segments.len() > 1 {
            segments[1..]
                .iter()
                .map(|s| s.parse())
                .collect::<TCResult<Vec<PathSegment>>>()?
                .try_into()?
        } else {
            TCPath::default()
        };

        Ok(Link {
            host: Some(host),
            path,
        })
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

impl<'de> serde::Deserialize<'de> for Link {
    fn deserialize<D>(deserializer: D) -> Result<Link, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        let m: HashMap<String, HashMap<ValueId, Value>> =
            de::Deserialize::deserialize(deserializer)?;
        if m.len() != 1 {
            Err(de::Error::custom(format!(
                "Expected Link, found Map: {}",
                m.keys().cloned().collect::<Vec<String>>().join(", ")
            )))
        } else {
            let mut m: Vec<_> = m.into_iter().collect();
            let (link, params) = m.pop().unwrap();
            if !params.is_empty() {
                Err(de::Error::custom(format!(
                    "Expected Link but found Op to {}",
                    link
                )))
            } else {
                link.parse().map_err(de::Error::custom)
            }
        }
    }
}

impl serde::Serialize for Link {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = s.serialize_map(Some(1))?;
        let data: HashMap<ValueId, Value> = HashMap::new();
        map.serialize_entry(&self.to_string(), &data)?;
        map.end()
    }
}

#[derive(Clone, Debug, Default, Hash, Eq, PartialEq)]
pub struct TCPath {
    segments: Vec<PathSegment>,
}

impl TCPath {
    pub fn from_path(&self, other: &TCPath) -> TCResult<TCPath> {
        if other.len() > self.len() {
            Err(error::bad_request(
                &format!("Expected {}/..., found", other),
                self,
            ))
        } else if self.segments[..other.len()] == other.segments[..] {
            Ok(TCPath {
                segments: self.segments[other.len()..].to_vec(),
            })
        } else {
            Err(error::bad_request(
                &format!(
                    "Tried to calculate the path of {} from an incorrect root",
                    self
                ),
                other,
            ))
        }
    }

    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    pub fn len(&self) -> usize {
        self.segments.len()
    }

    pub fn join(self, other: TCPath) -> TCPath {
        let mut segments = self.segments;
        segments.extend(other.segments);
        TCPath { segments }
    }

    pub fn pop(&mut self) -> Option<PathSegment> {
        self.segments.pop()
    }

    pub fn slice_from(&self, start: usize) -> TCPath {
        assert!(start <= self.segments.len());

        if start == self.segments.len() {
            TCPath { segments: vec![] }
        } else {
            TCPath {
                segments: self.segments[start..].to_vec(),
            }
        }
    }

    pub fn slice_to(&self, end: usize) -> TCPath {
        TCPath {
            segments: self.segments[..end].to_vec(),
        }
    }

    pub fn starts_with(&self, other: &TCPath) -> bool {
        if self.len() < other.len() {
            false
        } else {
            self.segments[0..other.len()] == other.segments[..]
        }
    }

    pub fn push(&mut self, segment: PathSegment) {
        self.segments.push(segment)
    }
}

impl Extend<PathSegment> for TCPath {
    fn extend<T: IntoIterator<Item = PathSegment>>(&mut self, iter: T) {
        self.segments.extend(iter)
    }
}

impl IntoIterator for TCPath {
    type Item = PathSegment;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.segments.into_iter()
    }
}

impl<Idx> std::ops::Index<Idx> for TCPath
where
    Idx: std::slice::SliceIndex<[PathSegment]>,
{
    type Output = Idx::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.segments[index]
    }
}

impl fmt::Display for TCPath {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "/{}",
            self.segments
                .iter()
                .map(String::from)
                .collect::<Vec<String>>()
                .join("/")
        )
    }
}

impl PartialEq<str> for TCPath {
    #[allow(clippy::cmp_owned)]
    fn eq(&self, other: &str) -> bool {
        self.to_string() == other
    }
}

impl PartialEq<ValueId> for TCPath {
    fn eq(&self, other: &ValueId) -> bool {
        if self.len() == 1 {
            &self.segments[0] == other
        } else {
            false
        }
    }
}

impl PartialEq<TCPath> for ValueId {
    fn eq(&self, other: &TCPath) -> bool {
        if other.len() == 1 {
            &other.segments[0] == self
        } else {
            false
        }
    }
}

impl From<Vec<PathSegment>> for TCPath {
    fn from(segments: Vec<PathSegment>) -> TCPath {
        TCPath { segments }
    }
}

impl From<PathSegment> for TCPath {
    fn from(segment: PathSegment) -> TCPath {
        TCPath {
            segments: vec![segment],
        }
    }
}

impl From<Label> for TCPath {
    fn from(segment: Label) -> TCPath {
        TCPath {
            segments: vec![segment.into()],
        }
    }
}

impl From<TCPath> for PathBuf {
    fn from(path: TCPath) -> PathBuf {
        PathBuf::from(format!("{}", path))
    }
}

impl FromStr for TCPath {
    type Err = error::TCError;

    fn from_str(to: &str) -> TCResult<TCPath> {
        if to == "/" {
            Ok(TCPath { segments: vec![] })
        } else if to.ends_with('/') {
            Err(error::bad_request("Path cannot end with a slash", to))
        } else if to.starts_with('/') {
            let mut segments: Vec<&str> = to.split('/').collect();
            segments.remove(0);
            let segments = segments
                .into_iter()
                .map(PathSegment::from_str)
                .collect::<TCResult<Vec<PathSegment>>>()?;
            Ok(TCPath { segments })
        } else {
            Ok(TCPath {
                segments: vec![to.parse()?],
            })
        }
    }
}

impl<'de> serde::Deserialize<'de> for TCPath {
    fn deserialize<D>(deserializer: D) -> Result<TCPath, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        let s: &str = de::Deserialize::deserialize(deserializer)?;
        s.parse().map_err(de::Error::custom)
    }
}

impl serde::Serialize for TCPath {
    fn serialize<S>(&self, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        s.serialize_str(&self.to_string())
    }
}
