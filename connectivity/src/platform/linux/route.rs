//! Reachability from rtnetlink: a socket subscribed to link and route
//! changes says when to look, a route dump says what is there.

use std::io;

use async_io::Async;
use netlink_sys::{Socket, SocketAddr, protocols::NETLINK_ROUTE};

use super::decode;
use crate::{ConnectivityError, hub::Sink};

/// `RTMGRP_LINK`, `RTMGRP_IPV4_ROUTE` and `RTMGRP_IPV6_ROUTE`, from
/// `linux/rtnetlink.h`.
const GROUPS: u32 = 0x1 | 0x40 | 0x400;

pub struct Watcher {
    events: Async<Socket>,
    dump: Async<Socket>,
}

impl Watcher {
    /// Opens the sockets and reports the initial reachability to `sink`.
    pub fn open(sink: &Sink) -> Result<Self, ConnectivityError> {
        let register = |reason: io::Error| ConnectivityError::Register {
            monitor: "rtnetlink",
            reason: reason.to_string(),
        };
        let mut events = Socket::new(NETLINK_ROUTE).map_err(register)?;
        events.bind(&SocketAddr::new(0, GROUPS)).map_err(register)?;
        events.set_non_blocking(true).map_err(register)?;
        let mut dump = Socket::new(NETLINK_ROUTE).map_err(register)?;
        dump.bind_auto().map_err(register)?;
        sink.set_online(has_default_route_sync(&dump).map_err(register)?);
        dump.set_non_blocking(true).map_err(register)?;
        Ok(Self {
            events: Async::new(events).map_err(register)?,
            dump: Async::new(dump).map_err(register)?,
        })
    }

    /// Re-reads the routes whenever the kernel announces a link or route
    /// change, until the socket fails.
    pub async fn watch(self, sink: &Sink) -> io::Result<()> {
        loop {
            match self
                .events
                .read_with(netlink_sys::Socket::recv_from_full)
                .await
            {
                Ok(_) => {}
                // The socket's receive buffer overran, so announcements were
                // lost; the dump below is what resynchronises.
                Err(error) if error.raw_os_error() == Some(libc::ENOBUFS) => {}
                Err(error) => {
                    tracing::error!(%error, "rtnetlink socket failed, reachability is no longer tracked");
                    return Err(error);
                }
            }
            // One change usually arrives as a burst; read the dump once.
            while self.events.get_ref().recv_from_full().is_ok() {}
            match has_default_route(&self.dump).await {
                Ok(online) => sink.set_online(online),
                Err(error) => {
                    tracing::error!(%error, "route dump failed, reachability is no longer tracked");
                    return Err(error);
                }
            }
        }
    }
}

struct DumpAccumulator {
    default_route: bool,
}

impl DumpAccumulator {
    const fn new() -> Self {
        Self {
            default_route: false,
        }
    }

    fn feed(&mut self, datagram: &[u8]) -> io::Result<Option<bool>> {
        let scan = decode::scan(datagram).map_err(io::Error::other)?;
        self.default_route |= scan.default_route;
        Ok(scan.done.then_some(self.default_route))
    }
}

/// Dumps every route synchronously for initial startup before entering the async loop.
fn has_default_route_sync(dump: &Socket) -> io::Result<bool> {
    let request = decode::dump_request();
    let sent = dump.send_to(&request, &SocketAddr::new(0, 0), 0)?;
    if sent != request.len() {
        return Err(io::Error::new(
            io::ErrorKind::WriteZero,
            "short netlink dump request write",
        ));
    }
    let mut acc = DumpAccumulator::new();
    loop {
        let (datagram, _) = dump.recv_from_full()?;
        if let Some(default_route) = acc.feed(&datagram)? {
            return Ok(default_route);
        }
    }
}

/// Dumps every route asynchronously and reports whether one is a default route.
async fn has_default_route(dump: &Async<Socket>) -> io::Result<bool> {
    let request = decode::dump_request();
    let kernel = SocketAddr::new(0, 0);
    let sent = dump
        .write_with(|sock| sock.send_to(&request, &kernel, 0))
        .await?;
    if sent != request.len() {
        return Err(io::Error::new(
            io::ErrorKind::WriteZero,
            "short netlink dump request write",
        ));
    }
    let mut acc = DumpAccumulator::new();
    loop {
        let (datagram, _) = dump.read_with(netlink_sys::Socket::recv_from_full).await?;
        if let Some(default_route) = acc.feed(&datagram)? {
            return Ok(default_route);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn async_route_dump_succeeds_without_blocking() {
        let mut dump = Socket::new(NETLINK_ROUTE).expect("open netlink route socket");
        dump.bind_auto().expect("bind dump socket");
        dump.set_non_blocking(true).expect("set non-blocking");
        let dump = Async::new(dump).expect("wrap in Async");

        let _online = has_default_route(&dump)
            .await
            .expect("async netlink route dump should succeed");
    }
}
