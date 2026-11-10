//! Where the host is reachable from inside a container.

use std::net::IpAddr;

/// Why an address and name cannot be a [`HostRoute`].
#[derive(Debug, thiserror::Error)]
pub enum HostRouteError {
    /// The name the container dials is blank.
    #[error("a host route requires a non-empty name")]
    BlankName,
    /// The unspecified wildcard address would listen on every interface.
    #[error("the unspecified address {0} would expose a host listener on every interface")]
    Unspecified(IpAddr),
    /// A multicast address cannot serve a listener.
    #[error("the multicast address {0} cannot serve a host listener")]
    Multicast(IpAddr),
}

/// How a container reaches the host: the `name` it dials and the host
/// `address` a listener meant for the container binds.
///
/// A listener the host opens for a container (an IPC bridge, say) executes
/// work for whoever connects, so it binds only an address the container
/// network alone reaches, never the unspecified (`0.0.0.0`, `::`) address,
/// which would expose it on every interface including the LAN, nor a
/// multicast address, which cannot serve a listener. Docker Desktop and
/// `OrbStack` forward `host.docker.internal` to the host's loopback, so the
/// address is `127.0.0.1` there; native Linux Docker routes the name to the
/// container network's gateway the host owns (`docker0` or the docker
/// `bridge` network gateway, e.g. `172.17.0.1`), so the address is that one.
/// Podman's name is `host.containers.internal`. The two belong to the same
/// route, and only the caller, which knows the container runtime, can pair
/// them correctly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostRoute {
    name: String,
    address: IpAddr,
}

impl HostRoute {
    /// The route a container uses to reach the host.
    ///
    /// # Errors
    /// Returns an error when `name` is blank or `address` is unspecified or
    /// multicast.
    pub fn new(name: impl Into<String>, address: IpAddr) -> Result<Self, HostRouteError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(HostRouteError::BlankName);
        }
        if address.is_unspecified() {
            return Err(HostRouteError::Unspecified(address));
        }
        if address.is_multicast() {
            return Err(HostRouteError::Multicast(address));
        }
        Ok(Self { name, address })
    }

    /// The name the container dials to reach the host.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The host address a listener meant for the container binds.
    #[must_use]
    pub const fn address(&self) -> IpAddr {
        self.address
    }
}

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

    use super::*;

    #[test]
    fn a_route_rejects_the_unspecified_and_multicast_addresses() {
        for address in [
            IpAddr::V4(Ipv4Addr::UNSPECIFIED),
            IpAddr::V6(Ipv6Addr::UNSPECIFIED),
        ] {
            assert!(matches!(
                HostRoute::new("host.docker.internal", address),
                Err(HostRouteError::Unspecified(_))
            ));
        }
        for address in [
            IpAddr::V4(Ipv4Addr::new(224, 0, 0, 1)),
            IpAddr::V6(Ipv6Addr::new(0xff02, 0, 0, 0, 0, 0, 0, 1)),
        ] {
            assert!(matches!(
                HostRoute::new("host.docker.internal", address),
                Err(HostRouteError::Multicast(_))
            ));
        }
    }

    #[test]
    fn a_route_needs_a_name() {
        assert!(matches!(
            HostRoute::new("  ", IpAddr::V4(Ipv4Addr::LOCALHOST)),
            Err(HostRouteError::BlankName)
        ));
    }

    #[test]
    fn a_route_names_what_the_container_dials_and_the_host_binds() {
        let route =
            HostRoute::new("host.docker.internal", IpAddr::V4(Ipv4Addr::LOCALHOST)).unwrap();
        assert_eq!(route.name(), "host.docker.internal");
        assert_eq!(route.address(), IpAddr::V4(Ipv4Addr::LOCALHOST));
    }

    #[test]
    fn loopback_and_gateway_addresses_are_accepted() {
        for address in [
            IpAddr::V4(Ipv4Addr::LOCALHOST),
            IpAddr::V6(Ipv6Addr::LOCALHOST),
            IpAddr::V4(Ipv4Addr::new(172, 17, 0, 1)),
        ] {
            HostRoute::new("host.docker.internal", address).unwrap();
        }
    }
}
