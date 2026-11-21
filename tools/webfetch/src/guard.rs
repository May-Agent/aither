//! Destination checks that keep a fetch pointed at the public internet.
//!
//! The URL this crate fetches is typically chosen by a language model, which
//! may be acting on text from a page it fetched a moment ago. That makes a
//! plain `GET` a server-side request forgery primitive: a model that is merely
//! confused — never mind one that has been prompt-injected — can be talked into
//! fetching `http://169.254.169.254/latest/meta-data/iam/security-credentials/`
//! and handing the result back as tool output.
//!
//! Every entry point therefore resolves the host and refuses any address that
//! is not publicly routable.
//!
//! Redirects are not a hole in the HTTP client, which is built without
//! redirect-following middleware, so a `302` is returned rather than followed;
//! if that ever changes, each hop must be re-checked through [`guard_url`].
//! The headless browser does follow redirects and load subresources, so it
//! reaches the network only through a forward proxy of its own (the `proxy`
//! module, behind the `headless` feature) that resolves each host once,
//! judges every address it resolved by this same policy, and connects to a
//! judged address only; the browser never resolves a destination itself.

use std::future::Future;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::Poll;
use std::time::Duration;

use futures_lite::io::{AsyncReadExt, AsyncWriteExt};
use url::{Host, Url};
use zenwave::{Client as _, DefaultClient, Proxy, ResponseExt as _, Transport, header};

/// Why a URL was refused before any request was made.
#[derive(Debug, thiserror::Error)]
pub enum UrlPolicyError {
    /// The string could not be parsed as a URL.
    #[error("invalid URL: {0}")]
    Invalid(#[from] url::ParseError),

    /// Only `http` and `https` are fetchable.
    #[error("unsupported URL scheme '{0}': only http and https are allowed")]
    UnsupportedScheme(String),

    /// The URL had no host component (for example `http:///foo`).
    #[error("URL has no host")]
    MissingHost,

    /// The host could not be resolved to any address.
    #[error("could not resolve host '{host}': {source}")]
    Resolve {
        /// The host that failed to resolve.
        host: String,
        /// The underlying resolver error.
        source: std::io::Error,
    },

    /// The host resolved to an address that is not publicly routable.
    #[error(
        "refusing to fetch '{host}': it resolves to {addr}, which is a private, loopback, or \
         link-local address"
    )]
    BlockedAddress {
        /// The host as written in the URL.
        host: String,
        /// The disallowed address it resolved to.
        addr: IpAddr,
    },
}

/// Whether an address is one this crate refuses to connect to.
///
/// Covers loopback, unspecified, multicast and broadcast addresses, the RFC1918
/// private ranges, link-local (including the cloud metadata range), carrier-grade
/// NAT, IPv6 unique-local and link-local, and IPv4-mapped forms of all of these.
#[must_use]
pub const fn is_blocked(addr: IpAddr) -> bool {
    match addr {
        IpAddr::V4(v4) => is_blocked_v4(v4),
        IpAddr::V6(v6) => {
            // An IPv4-mapped address reaches the same host as the bare IPv4
            // address, so it must be judged by the same rules.
            if let Some(mapped) = v6.to_ipv4_mapped() {
                return is_blocked_v4(mapped);
            }
            is_blocked_v6(v6)
        }
    }
}

const fn is_blocked_v4(addr: Ipv4Addr) -> bool {
    addr.is_loopback()
        || addr.is_private()
        || addr.is_link_local()
        || addr.is_broadcast()
        || addr.is_multicast()
        || addr.is_unspecified()
        || addr.is_documentation()
        // 100.64.0.0/10, carrier-grade NAT (Ipv4Addr::is_shared is unstable).
        || (addr.octets()[0] == 100 && (addr.octets()[1] & 0b1100_0000) == 64)
        // 192.0.0.0/24, IETF protocol assignments.
        || matches!(addr.octets(), [192, 0, 0, _])
        // 240.0.0.0/4, reserved.
        || addr.octets()[0] >= 240
}

const fn is_blocked_v6(addr: Ipv6Addr) -> bool {
    let segments = addr.segments();
    // 64:ff9b:1::/48, RFC 8215 local-use IPv4/IPv6 translation prefix.
    if segments[0] == 0x0064 && segments[1] == 0xff9b && segments[2] == 0x0001 {
        return true;
    }
    // 64:ff9b::/96, RFC 6052 well-known IPv4/IPv6 translation prefix.
    if segments[0] == 0x0064
        && segments[1] == 0xff9b
        && segments[2] == 0
        && segments[3] == 0
        && segments[4] == 0
        && segments[5] == 0
    {
        let [a, b] = segments[6].to_be_bytes();
        let [c, d] = segments[7].to_be_bytes();
        if is_blocked_v4(Ipv4Addr::new(a, b, c, d)) {
            return true;
        }
    }
    // 2002::/16, RFC 3056 6to4 prefix carrying an embedded IPv4 in bits 16..47.
    if segments[0] == 0x2002 {
        let [a, b] = segments[1].to_be_bytes();
        let [c, d] = segments[2].to_be_bytes();
        if is_blocked_v4(Ipv4Addr::new(a, b, c, d)) {
            return true;
        }
    }

    addr.is_loopback()
        || addr.is_unspecified()
        || addr.is_multicast()
        // fc00::/7, unique local (Ipv6Addr::is_unique_local is unstable).
        || (segments[0] & 0xfe00) == 0xfc00
        // fe80::/10, link-local unicast.
        || (segments[0] & 0xffc0) == 0xfe80
}

/// Parses `raw` and confirms it points somewhere publicly routable.
///
/// # Errors
///
/// Returns [`UrlPolicyError`] if the URL is malformed, uses a scheme other than
/// http/https, has no host, cannot be resolved, or resolves to any address that
/// [`is_blocked`] rejects. A host resolving to several addresses is refused if
/// *any* of them is blocked, so a split-horizon DNS answer cannot smuggle one
/// past.
pub async fn guard_url(raw: &str) -> Result<Url, UrlPolicyError> {
    let url = Url::parse(raw)?;

    let scheme = url.scheme();
    if scheme != "http" && scheme != "https" {
        return Err(UrlPolicyError::UnsupportedScheme(scheme.to_string()));
    }

    let host = url.host().ok_or(UrlPolicyError::MissingHost)?;
    // Every scheme left has a default port.
    let port = url.port_or_known_default().unwrap_or(80);
    resolve_checked(&SystemResolver, &PublicOnly, &host, port).await?;

    Ok(url)
}

/// Resolves host names to socket addresses.
pub trait Resolver: Clone + Send + Sync + 'static {
    /// Every address `host` answers to on `port`.
    ///
    /// # Errors
    ///
    /// Returns the resolver's error when `host` cannot be resolved.
    fn resolve(
        &self,
        host: &str,
        port: u16,
    ) -> impl Future<Output = std::io::Result<Vec<SocketAddr>>> + Send;
}

/// The operating system's resolver.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemResolver;

impl Resolver for SystemResolver {
    async fn resolve(&self, host: &str, port: u16) -> std::io::Result<Vec<SocketAddr>> {
        async_net::resolve((host, port)).await
    }
}

/// Decides which addresses a fetch may connect to.
pub trait AddressPolicy: Clone + Send + Sync + 'static {
    /// Judges `addr`, which `host` resolved to.
    ///
    /// # Errors
    ///
    /// Returns why nothing may connect to `addr`.
    fn check(&self, host: &str, addr: SocketAddr) -> Result<(), UrlPolicyError>;
}

/// The policy of every fetch: only publicly routable addresses, as
/// [`is_blocked`] decides.
#[derive(Debug, Clone, Copy, Default)]
pub struct PublicOnly;

impl AddressPolicy for PublicOnly {
    fn check(&self, host: &str, addr: SocketAddr) -> Result<(), UrlPolicyError> {
        if is_blocked(addr.ip()) {
            return Err(UrlPolicyError::BlockedAddress {
                host: host.to_string(),
                addr: addr.ip(),
            });
        }
        Ok(())
    }
}

/// Resolves `host` once and returns the addresses to connect to, every one of
/// them approved by `policy`.
///
/// An IP literal is judged as it is, without consulting `resolver`. A host is
/// refused if *any* address it resolved to is refused, so a split-horizon
/// answer cannot smuggle one past.
///
/// # Errors
///
/// Returns [`UrlPolicyError::Resolve`] when `host` resolves to nothing, and
/// the policy's refusal of the first address it refuses.
pub async fn resolve_checked<R: Resolver, P: AddressPolicy>(
    resolver: &R,
    policy: &P,
    host: &Host<&str>,
    port: u16,
) -> Result<Vec<SocketAddr>, UrlPolicyError> {
    let host_str = host.to_string();
    let addrs =
        match *host {
            Host::Ipv4(addr) => vec![SocketAddr::new(IpAddr::V4(addr), port)],
            Host::Ipv6(addr) => vec![SocketAddr::new(IpAddr::V6(addr), port)],
            Host::Domain(domain) => {
                let addrs = resolver.resolve(domain, port).await.map_err(|source| {
                    UrlPolicyError::Resolve {
                        host: host_str.clone(),
                        source,
                    }
                })?;
                if addrs.is_empty() {
                    return Err(UrlPolicyError::Resolve {
                        host: host_str,
                        source: std::io::Error::new(
                            std::io::ErrorKind::NotFound,
                            "host resolved to no addresses",
                        ),
                    });
                }
                addrs
            }
        };
    for addr in &addrs {
        policy.check(&host_str, *addr)?;
    }
    Ok(addrs)
}

const MAX_REDIRECTS: usize = 10;
const MAX_HTTP_BODY_BYTES: usize = 32 * 1024 * 1024;

pub(crate) struct GuardedHttpResponse {
    pub headers: header::HeaderMap,
    pub body: Vec<u8>,
}

type ClientTask = Pin<Box<dyn Future<Output = ()> + Send>>;

async fn run_socks5_gate<R, P>(
    listener: async_net::TcpListener,
    resolver: R,
    policy: P,
    refusal: Arc<Mutex<Option<UrlPolicyError>>>,
) -> Result<GuardedHttpResponse, GuardedGetError>
where
    R: Resolver,
    P: AddressPolicy,
{
    use futures_lite::StreamExt as _;
    let mut incoming = Box::pin(listener.incoming());
    let mut active: Vec<ClientTask> = Vec::new();
    std::future::poll_fn(|cx| {
        while let Poll::Ready(Some(Ok(stream))) = incoming.as_mut().poll_next(cx) {
            let resolver = resolver.clone();
            let policy = policy.clone();
            let refusal = Arc::clone(&refusal);
            active.push(Box::pin(async move {
                if let Err(policy_err) = handle_socks5_client(stream, resolver, policy).await {
                    let mut slot = refusal
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    if slot.is_none() {
                        *slot = Some(policy_err);
                    }
                }
            }));
        }
        active.retain_mut(|fut| fut.as_mut().poll(cx).is_pending());
        Poll::<Result<GuardedHttpResponse, GuardedGetError>>::Pending
    })
    .await
}

/// Performs an HTTP `GET` where every hop (initial URL and any HTTP redirects)
/// is checked against `resolver` and `policy`, the outbound TCP connection
/// dials ONLY the validated [`SocketAddr`]s (preventing DNS rebinding and
/// redirect-to-private SSRF), and the response body is read while the gate is
/// active.
pub(crate) async fn guarded_get<R, P>(
    raw_url: &str,
    headers: &[(&str, String)],
    timeout: Duration,
    resolver: R,
    policy: P,
) -> Result<GuardedHttpResponse, GuardedGetError>
where
    R: Resolver,
    P: AddressPolicy,
{
    let initial = parse_http_url(raw_url)?;
    let listener = async_net::TcpListener::bind("127.0.0.1:0").await?;
    let gate_port = listener.local_addr()?.port();
    let refusal: Arc<Mutex<Option<UrlPolicyError>>> = Arc::new(Mutex::new(None));

    let proxy = Proxy::builder()
        .all(format!("socks5h://127.0.0.1:{gate_port}"))
        .build();
    let transport = Transport::builder()
        .proxy(proxy)
        .build()
        .map_err(|e| GuardedGetError::Transport(e.into()))?;

    let gate_future = run_socks5_gate(listener, resolver, policy, Arc::clone(&refusal));

    let request_future = async {
        let mut current = initial;
        let initial_origin = current.origin();
        for _hop in 0..=MAX_REDIRECTS {
            let mut backend = DefaultClient::raw(transport.clone()).timeout(timeout);
            let mut builder = backend
                .get(current.as_str())
                .map_err(|e| GuardedGetError::Transport(e.into()))?;
            let same_origin = current.origin() == initial_origin;
            for (name, value) in headers {
                if !same_origin
                    && (name.eq_ignore_ascii_case("authorization")
                        || name.eq_ignore_ascii_case("x-api-key")
                        || name.eq_ignore_ascii_case("cookie"))
                {
                    continue;
                }
                builder = builder
                    .header(*name, value.clone())
                    .map_err(|e| GuardedGetError::Transport(e.into()))?;
            }
            let response = match builder.await {
                Ok(resp) => resp,
                Err(err) => {
                    let taken_refusal = refusal
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .take();
                    if let Some(policy_err) = taken_refusal {
                        return Err(policy_err.into());
                    }
                    return Err(GuardedGetError::Transport(err.into()));
                }
            };

            let code = response.status().as_u16();
            if matches!(code, 301 | 302 | 303 | 307 | 308)
                && let Some(loc) = response
                    .headers()
                    .get(header::LOCATION.as_str())
                    .and_then(|v| v.to_str().ok())
            {
                let next = current.join(loc).map_err(UrlPolicyError::Invalid)?;
                current = parse_http_url(next.as_str())?;
                continue;
            }
            let response = response
                .error_for_status()
                .await
                .map_err(|err| GuardedGetError::Transport(err.into()))?;
            let resp_headers = response.headers().clone();
            let bytes = response
                .into_bytes_with_limit(MAX_HTTP_BODY_BYTES)
                .await
                .map_err(|e| GuardedGetError::from(DecodeFailure(e.to_string())))?;
            return Ok(GuardedHttpResponse {
                headers: resp_headers,
                body: bytes.to_vec(),
            });
        }
        Err(GuardedGetError::RedirectLimit)
    };

    futures_lite::future::or(request_future, gate_future).await
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub(crate) struct DecodeFailure(pub String);

/// Why a guarded HTTP `GET` failed.
#[derive(Debug, thiserror::Error)]
pub(crate) enum GuardedGetError {
    /// The URL policy refused the target.
    #[error("{0}")]
    Policy(#[from] UrlPolicyError),
    /// The local gate or an I/O step failed.
    #[error("{0}")]
    Io(#[from] std::io::Error),
    /// Decoding the response body failed.
    #[error("{0}")]
    Decode(#[from] DecodeFailure),
    /// The HTTP transport or the request itself failed.
    #[error("{0}")]
    Transport(#[source] Box<dyn std::error::Error + Send + Sync>),
    /// The redirect chain exceeded the limit.
    #[error("too many HTTP redirects")]
    RedirectLimit,
}

fn parse_http_url(raw: &str) -> Result<Url, UrlPolicyError> {
    let url = Url::parse(raw)?;
    let scheme = url.scheme();
    if scheme != "http" && scheme != "https" {
        return Err(UrlPolicyError::UnsupportedScheme(scheme.to_string()));
    }
    if url.host().is_none() {
        return Err(UrlPolicyError::MissingHost);
    }
    Ok(url)
}

async fn read_socks5_target(client: &mut async_net::TcpStream) -> Option<(String, u16)> {
    let mut greeting = [0_u8; 2];
    if client.read_exact(&mut greeting).await.is_err() || greeting[0] != 0x05 {
        return None;
    }
    let mut methods = vec![0_u8; usize::from(greeting[1])];
    if client.read_exact(&mut methods).await.is_err() {
        return None;
    }
    if client.write_all(&[0x05, 0x00]).await.is_err() {
        return None;
    }

    let mut head = [0_u8; 4];
    if client.read_exact(&mut head).await.is_err() || head[0] != 0x05 || head[1] != 0x01 {
        let _ = client
            .write_all(&[0x05, 0x07, 0x00, 0x01, 0, 0, 0, 0, 0, 0])
            .await;
        return None;
    }

    let host_string = match head[3] {
        0x01 => {
            let mut octets = [0_u8; 4];
            client.read_exact(&mut octets).await.ok()?;
            Ipv4Addr::from(octets).to_string()
        }
        0x04 => {
            let mut octets = [0_u8; 16];
            client.read_exact(&mut octets).await.ok()?;
            format!("[{}]", Ipv6Addr::from(octets))
        }
        0x03 => {
            let mut len = [0_u8; 1];
            client.read_exact(&mut len).await.ok()?;
            let mut domain = vec![0_u8; usize::from(len[0])];
            client.read_exact(&mut domain).await.ok()?;
            let Ok(text) = String::from_utf8(domain) else {
                let _ = client
                    .write_all(&[0x05, 0x04, 0x00, 0x01, 0, 0, 0, 0, 0, 0])
                    .await;
                return None;
            };
            text
        }
        _ => {
            let _ = client
                .write_all(&[0x05, 0x08, 0x00, 0x01, 0, 0, 0, 0, 0, 0])
                .await;
            return None;
        }
    };

    let mut port_bytes = [0_u8; 2];
    client.read_exact(&mut port_bytes).await.ok()?;
    Some((host_string, u16::from_be_bytes(port_bytes)))
}

async fn handle_socks5_client<R, P>(
    mut client: async_net::TcpStream,
    resolver: R,
    policy: P,
) -> Result<(), UrlPolicyError>
where
    R: Resolver,
    P: AddressPolicy,
{
    let Some((host_string, port)) = read_socks5_target(&mut client).await else {
        return Ok(());
    };

    let parsed_host = Host::parse(&host_string)?;
    let host_ref = match &parsed_host {
        Host::Domain(domain) => Host::Domain(domain.as_str()),
        Host::Ipv4(addr) => Host::Ipv4(*addr),
        Host::Ipv6(addr) => Host::Ipv6(*addr),
    };
    let addrs = match resolve_checked(&resolver, &policy, &host_ref, port).await {
        Ok(addrs) => addrs,
        Err(policy_err) => {
            let _ = client
                .write_all(&[0x05, 0x02, 0x00, 0x01, 0, 0, 0, 0, 0, 0])
                .await;
            return Err(policy_err);
        }
    };

    let mut upstream = None;
    for addr in &addrs {
        if let Ok(conn) = async_net::TcpStream::connect(*addr).await {
            upstream = Some(conn);
            break;
        }
    }
    let Some(upstream) = upstream else {
        let _ = client
            .write_all(&[0x05, 0x05, 0x00, 0x01, 0, 0, 0, 0, 0, 0])
            .await;
        return Ok(());
    };

    if client
        .write_all(&[0x05, 0x00, 0x00, 0x01, 0, 0, 0, 0, 0, 0])
        .await
        .is_err()
    {
        return Ok(());
    }

    let (client_r, client_w) = futures_lite::io::split(client);
    let (up_r, up_w) = futures_lite::io::split(upstream);
    let _ = futures_lite::future::zip(
        futures_lite::io::copy(client_r, up_w),
        futures_lite::io::copy(up_r, client_w),
    )
    .await;
    Ok(())
}

/// A resolver and a policy tests control.
#[cfg(test)]
pub(crate) mod testing {
    use super::{AddressPolicy, Resolver, UrlPolicyError};
    use std::collections::HashMap;
    use std::future::Future;
    use std::net::SocketAddr;
    use std::sync::{Arc, Mutex};

    /// Answers lookups from a fixed table and records them; a host listed
    /// several times answers with every address listed for it.
    #[derive(Clone, Default)]
    pub struct Table {
        hosts: Arc<HashMap<String, Vec<SocketAddr>>>,
        asked: Arc<Mutex<Vec<String>>>,
    }

    impl Table {
        pub(crate) fn new(entries: &[(&str, SocketAddr)]) -> Self {
            let mut hosts: HashMap<String, Vec<SocketAddr>> = HashMap::new();
            for (host, addr) in entries {
                hosts.entry((*host).to_string()).or_default().push(*addr);
            }
            Self {
                hosts: Arc::new(hosts),
                asked: Arc::default(),
            }
        }

        /// Every host the table was asked for, in order.
        pub(crate) fn asked(&self) -> Vec<String> {
            self.asked.lock().expect("lookup log poisoned").clone()
        }

        /// How many times the table was asked for `host`.
        pub(crate) fn lookups(&self, host: &str) -> usize {
            self.asked().iter().filter(|asked| *asked == host).count()
        }
    }

    impl Resolver for Table {
        fn resolve(
            &self,
            host: &str,
            _port: u16,
        ) -> impl Future<Output = std::io::Result<Vec<SocketAddr>>> + Send {
            self.asked
                .lock()
                .expect("lookup log poisoned")
                .push(host.to_string());
            let result = self.hosts.get(host).cloned().ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::NotFound, "not in the table")
            });
            std::future::ready(result)
        }
    }

    /// Allows exactly one address, which stands in for a public one.
    #[derive(Clone)]
    pub struct Only(pub(crate) SocketAddr);

    impl AddressPolicy for Only {
        fn check(&self, host: &str, addr: SocketAddr) -> Result<(), UrlPolicyError> {
            if addr == self.0 {
                Ok(())
            } else {
                Err(UrlPolicyError::BlockedAddress {
                    host: host.to_string(),
                    addr: addr.ip(),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::Table;
    use super::{
        AddressPolicy, PublicOnly, UrlPolicyError, guard_url, is_blocked, resolve_checked,
    };
    use std::net::{IpAddr, SocketAddr};
    use url::Host;

    fn sock(s: &str) -> SocketAddr {
        s.parse().expect("test socket address should parse")
    }

    #[tokio::test]
    async fn a_host_is_refused_when_any_of_its_addresses_is() {
        let resolver = Table::new(&[
            ("split.test", sock("93.184.216.34:80")),
            ("split.test", sock("10.0.0.5:80")),
        ]);
        let err = resolve_checked(&resolver, &PublicOnly, &Host::Domain("split.test"), 80)
            .await
            .unwrap_err();
        assert!(
            matches!(err, UrlPolicyError::BlockedAddress { addr, .. } if addr == ip("10.0.0.5"))
        );
    }

    #[tokio::test]
    async fn the_checked_addresses_are_the_ones_returned() {
        let resolver = Table::new(&[("ok.test", sock("93.184.216.34:443"))]);
        let addrs = resolve_checked(&resolver, &PublicOnly, &Host::Domain("ok.test"), 443)
            .await
            .unwrap();
        assert_eq!(addrs, vec![sock("93.184.216.34:443")]);
        assert_eq!(resolver.lookups("ok.test"), 1);
    }

    #[tokio::test]
    async fn an_unknown_host_is_a_resolve_error() {
        let err = resolve_checked(&Table::default(), &PublicOnly, &Host::Domain("x.test"), 80)
            .await
            .unwrap_err();
        assert!(matches!(err, UrlPolicyError::Resolve { .. }));
    }

    #[test]
    fn the_public_policy_judges_by_ip() {
        assert!(PublicOnly.check("a", sock("127.0.0.1:443")).is_err());
        assert!(PublicOnly.check("a", sock("1.1.1.1:443")).is_ok());
    }

    fn ip(s: &str) -> IpAddr {
        s.parse().expect("test address should parse")
    }

    #[test]
    fn blocks_loopback_and_private_ranges() {
        for addr in [
            "127.0.0.1",
            "127.13.2.9",
            "10.0.0.1",
            "172.16.0.1",
            "172.31.255.255",
            "192.168.1.1",
            "0.0.0.0",
            "255.255.255.255",
            "100.64.0.1",
            "::1",
            "::",
            "fd00::1",
            "fe80::1",
            "::ffff:127.0.0.1",
            "::ffff:10.0.0.1",
        ] {
            assert!(is_blocked(ip(addr)), "{addr} should be blocked");
        }
    }

    #[test]
    fn blocks_the_cloud_metadata_address() {
        assert!(is_blocked(ip("169.254.169.254")));
    }

    #[test]
    fn allows_public_addresses() {
        for addr in ["1.1.1.1", "8.8.8.8", "93.184.216.34", "2606:4700::1111"] {
            assert!(!is_blocked(ip(addr)), "{addr} should be allowed");
        }
    }

    #[tokio::test]
    async fn rejects_non_http_schemes() {
        let err = guard_url("file:///etc/passwd").await.unwrap_err();
        assert!(matches!(err, UrlPolicyError::UnsupportedScheme(_)));
    }

    #[tokio::test]
    async fn rejects_literal_metadata_url_without_dns() {
        let err = guard_url("http://169.254.169.254/latest/meta-data/")
            .await
            .unwrap_err();
        assert!(matches!(err, UrlPolicyError::BlockedAddress { .. }));
    }

    #[tokio::test]
    async fn rejects_localhost_literal() {
        let err = guard_url("http://127.0.0.1:8080/admin").await.unwrap_err();
        assert!(matches!(err, UrlPolicyError::BlockedAddress { .. }));
    }

    #[test]
    fn blocks_nat64_and_6to4_embedded_private_addresses() {
        for blocked in [
            "64:ff9b::127.0.0.1",
            "64:ff9b::169.254.169.254",
            "64:ff9b::10.0.0.1",
            "64:ff9b:1::1",
            "2002:7f00:0001::1",
            "2002:a9fe:a9fe::1",
        ] {
            assert!(is_blocked(ip(blocked)), "{blocked} should be blocked");
        }
        for public in ["64:ff9b::1.1.1.1", "2002:0101:0101::1"] {
            assert!(!is_blocked(ip(public)), "{public} should be allowed");
        }
    }

    #[tokio::test]
    async fn rejects_ipv6_loopback_literal() {
        let err = guard_url("http://[::1]/").await.unwrap_err();
        assert!(matches!(err, UrlPolicyError::BlockedAddress { .. }));
    }
}
