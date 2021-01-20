use std::collections::HashMap;
use std::convert::TryFrom;
use std::fmt;
use std::iter;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::ops::Deref;
use std::path::PathBuf;
use std::str::FromStr;

use addr::DomainName;
use async_trait::async_trait;
use destream::{de, Decoder, EncodeMap, Encoder, FromStream, ToStream};
use error::*;
use generic::{label, Id, Label};
use safecast::TryCastFrom;

use super::Value;

const EMPTY_SLICE: &[usize] = &[];

pub type PathSegment = Id;

pub struct PathLabel {
    segments: &'static [&'static str],
}

pub const fn path_label(segments: &'static [&'static str]) -> PathLabel {
    PathLabel { segments }
}

impl From<PathLabel> for TCPathBuf {
    fn from(path: PathLabel) -> Self {
        let segments = path
            .segments
            .into_iter()
            .map(|segment| label(*segment))
            .map(PathSegment::from)
            .collect();

        Self { segments }
    }
}

impl From<PathLabel> for Link {
    fn from(path: PathLabel) -> Self {
        TCPathBuf::from(path).into()
    }
}

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
            IPv4(addr) => IPv4(*addr),
            IPv6(addr) => IPv6(*addr),
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

fn is_ipv4(s: &str) -> bool {
    let segments: Vec<&str> = s.split('.').collect();
    segments.len() == 4 && !segments.iter().any(|s| s.parse::<u16>().is_err())
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

            let address: LinkAddress = if is_ipv4(address) {
                let address: Ipv4Addr = address
                    .parse()
                    .map_err(|e| TCError::bad_request("Unable to parse IPv4 address", e))?;

                address.into()
            } else {
                let address: DomainName = address
                    .parse()
                    .map_err(|e| TCError::bad_request("Unable to parse domain name", e))?;

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
    path: TCPathBuf,
}

impl Link {
    pub fn into_path(self) -> TCPathBuf {
        self.path
    }

    pub fn host(&'_ self) -> &'_ Option<LinkHost> {
        &self.host
    }

    pub fn path(&'_ self) -> &'_ TCPathBuf {
        &self.path
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
        } else if s.ends_with('/') {
            return Err(TCError::bad_request(
                "Link is not allowed to end with a '/'",
                s,
            ));
        }

        let segments: Vec<&str> = s.split('/').collect();
        if segments.is_empty() {
            return Err(TCError::bad_request("Unable to parse Link", s));
        }

        let host: LinkHost = segments[..3].join("/").parse()?;

        let segments = &segments[3..];

        let segments = segments
            .iter()
            .map(|s| s.parse())
            .collect::<TCResult<Vec<PathSegment>>>()?;

        Ok(Link {
            host: Some(host),
            path: TCPathBuf { segments },
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

#[async_trait]
impl FromStream for Link {
    async fn from_stream<D: Decoder>(decoder: &mut D) -> Result<Link, D::Error> {
        let m = HashMap::<String, HashMap<Id, Value>>::from_stream(decoder).await?;

        if m.len() != 1 {
            Err(de::Error::custom(format!(
                "Expected Link, found Map: {}",
                m.into_iter()
                    .map(|(k, _)| k)
                    .collect::<Vec<String>>()
                    .join(", ")
            )))
        } else {
            let mut m: Vec<_> = m.into_iter().collect();
            let (link, params) = m.pop().unwrap();
            if !params.is_empty() {
                Err(de::Error::custom(format!(
                    "Expected Link but found Op with subject {}",
                    link
                )))
            } else {
                link.parse().map_err(de::Error::custom)
            }
        }
    }
}

impl<'en> ToStream<'en> for Link {
    fn to_stream<E: Encoder<'en>>(&'en self, e: E) -> Result<E::Ok, E::Error> {
        let mut map = e.encode_map(Some(1))?;
        map.encode_entry(self.to_string(), &EMPTY_SLICE)?;
        map.end()
    }
}

#[derive(Clone, Debug, Default, Hash, Eq, PartialEq)]
pub struct TCPathBuf {
    segments: Vec<PathSegment>,
}

impl TCPathBuf {
    pub fn as_mut(&'_ mut self) -> &'_ mut Vec<PathSegment> {
        &mut self.segments
    }

    pub fn as_slice(&'_ self) -> &'_ [PathSegment] {
        &self.segments[..]
    }

    pub fn into_vec(self) -> Vec<PathSegment> {
        self.segments
    }

    pub fn append<T: Into<PathSegment>>(mut self, suffix: T) -> Self {
        self.segments.push(suffix.into());
        self
    }

    pub fn slice_from(self, index: usize) -> Self {
        TCPathBuf {
            segments: self.segments.into_iter().skip(index).collect(),
        }
    }

    pub fn suffix<'a>(&self, path: &'a [PathSegment]) -> Option<&'a [PathSegment]> {
        if path.starts_with(&self.segments) {
            Some(&path[self.segments.len()..])
        } else {
            None
        }
    }

    pub fn try_suffix<'a>(&self, path: &'a [PathSegment]) -> TCResult<&'a [PathSegment]> {
        self.suffix(path).ok_or_else(|| {
            TCError::internal(format!("{} routed through {}!", TCPath::from(path), self))
        })
    }
}

impl<Idx: std::slice::SliceIndex<[PathSegment]>> std::ops::Index<Idx> for TCPathBuf {
    type Output = Idx::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.segments[index]
    }
}

impl Extend<PathSegment> for TCPathBuf {
    fn extend<T: IntoIterator<Item = PathSegment>>(&mut self, iter: T) {
        self.segments.extend(iter)
    }
}

impl IntoIterator for TCPathBuf {
    type Item = PathSegment;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.segments.into_iter()
    }
}

impl std::borrow::Borrow<[PathSegment]> for TCPathBuf {
    fn borrow(&self) -> &[PathSegment] {
        &self.segments[..]
    }
}

impl Deref for TCPathBuf {
    type Target = [PathSegment];

    fn deref(&'_ self) -> &'_ [PathSegment] {
        &self.segments[..]
    }
}

impl From<PathSegment> for TCPathBuf {
    fn from(segment: PathSegment) -> TCPathBuf {
        TCPathBuf {
            segments: iter::once(segment).collect(),
        }
    }
}

impl From<Label> for TCPathBuf {
    fn from(segment: Label) -> TCPathBuf {
        TCPathBuf {
            segments: iter::once(segment.into()).collect(),
        }
    }
}

impl From<TCPathBuf> for PathBuf {
    fn from(path: TCPathBuf) -> PathBuf {
        PathBuf::from(path.to_string())
    }
}

impl FromStr for TCPathBuf {
    type Err = TCError;

    fn from_str(to: &str) -> TCResult<TCPathBuf> {
        if to == "/" {
            Ok(TCPathBuf { segments: vec![] })
        } else if to.ends_with('/') {
            Err(TCError::bad_request("Path cannot end with a slash", to))
        } else if to.starts_with('/') {
            let segments = to
                .split('/')
                .skip(1)
                .map(PathSegment::from_str)
                .collect::<TCResult<Vec<PathSegment>>>()?;

            Ok(TCPathBuf { segments })
        } else {
            Ok(TCPathBuf {
                segments: iter::once(to.parse()?).collect(),
            })
        }
    }
}

impl TryCastFrom<Value> for TCPathBuf {
    fn can_cast_from(value: &Value) -> bool {
        if let Value::String(s) = value {
            Self::from_str(s).is_ok()
        } else {
            false
        }
    }

    fn opt_cast_from(value: Value) -> Option<Self> {
        if let Value::String(s) = value {
            Self::from_str(&s).ok()
        } else {
            None
        }
    }
}

#[async_trait]
impl FromStream for TCPathBuf {
    async fn from_stream<D: Decoder>(decoder: &mut D) -> Result<TCPathBuf, D::Error> {
        let s = String::from_stream(decoder).await?;
        s.parse().map_err(de::Error::custom)
    }
}

impl<'en> ToStream<'en> for TCPathBuf {
    fn to_stream<E: Encoder<'en>>(&'en self, e: E) -> Result<E::Ok, E::Error> {
        e.encode_str(&self.to_string())
    }
}

impl TryCastFrom<TCPathBuf> for Id {
    fn can_cast_from(path: &TCPathBuf) -> bool {
        path.as_slice().len() == 1
    }

    fn opt_cast_from(path: TCPathBuf) -> Option<Id> {
        let mut segments = path.into_vec();
        if segments.len() == 1 {
            segments.pop()
        } else {
            None
        }
    }
}

impl fmt::Display for TCPathBuf {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", TCPath::from(&self[..]))
    }
}

pub struct TCPath<'a> {
    inner: &'a [PathSegment],
}

impl<'a> From<&'a [PathSegment]> for TCPath<'a> {
    fn from(inner: &'a [PathSegment]) -> TCPath<'a> {
        TCPath { inner }
    }
}

impl<'a, Idx: std::slice::SliceIndex<[PathSegment]>> std::ops::Index<Idx> for TCPath<'a> {
    type Output = Idx::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.inner[index]
    }
}

impl<'a> fmt::Display for TCPath<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "/{}",
            self.inner
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<String>>()
                .join("/")
        )
    }
}
