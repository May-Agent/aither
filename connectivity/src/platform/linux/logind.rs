//! Sleep and wake from logind's `PrepareForSleep(b)` signal on the system
//! bus: `true` just before the system suspends, `false` just after it
//! resumes.
//!
//! <https://www.freedesktop.org/software/systemd/man/latest/org.freedesktop.login1.html>

use futures_lite::StreamExt;
use zbus::connection::Builder;
use zbus::{Proxy, proxy::SignalStream};

use crate::{ConnectivityError, hub::Sink};

const SERVICE: &str = "org.freedesktop.login1";
const PATH: &str = "/org/freedesktop/login1";
const INTERFACE: &str = "org.freedesktop.login1.Manager";

pub struct Watcher {
    signals: SignalStream<'static>,
}

fn register(error: &zbus::Error) -> ConnectivityError {
    ConnectivityError::Register {
        monitor: "logind PrepareForSleep",
        reason: error.to_string(),
    }
}

/// Whether `error` is the bus socket not existing.
fn bus_is_absent(error: &zbus::Error) -> bool {
    matches!(error, zbus::Error::Connection(io, _) if io.kind() == std::io::ErrorKind::NotFound)
}

impl Watcher {
    /// Connects to the system bus and subscribes to `PrepareForSleep`.
    ///
    /// `None` when the machine has no system bus socket, as in a container
    /// or on a minimal server: nothing there suspends it through logind, so
    /// there are no sleeps to report. Any other failure to reach the bus is
    /// an error.
    pub async fn connect() -> Result<Option<Self>, ConnectivityError> {
        Self::connect_with(Builder::system()).await
    }

    async fn connect_with(builder: Builder<'_>) -> Result<Option<Self>, ConnectivityError> {
        let connection = match builder.build().await {
            Ok(connection) => connection,
            Err(error) if bus_is_absent(&error) => return Ok(None),
            Err(error) => return Err(register(&error)),
        };
        let proxy = Proxy::new(&connection, SERVICE, PATH, INTERFACE)
            .await
            .map_err(|error| register(&error))?;
        let signals = proxy
            .receive_signal("PrepareForSleep")
            .await
            .map_err(|error| register(&error))?;
        Ok(Some(Self { signals }))
    }

    /// Publishes every sleep and wake until the bus connection ends.
    pub async fn watch(mut self, sink: &Sink) {
        while let Some(message) = self.signals.next().await {
            match message.body().deserialize::<bool>() {
                Ok(true) => sink.sleeping(),
                Ok(false) => sink.woke(),
                Err(error) => tracing::warn!(%error, "malformed PrepareForSleep signal"),
            }
        }
        tracing::error!("system bus connection closed, sleep and wake are no longer tracked");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_missing_system_bus_means_no_sleep_events() {
        let builder = Builder::address("unix:path=/nonexistent/aither-system-bus");
        let watcher = Watcher::connect_with(builder)
            .await
            .expect("an absent bus is no error");
        assert!(watcher.is_none());
    }

    #[tokio::test]
    async fn a_bus_that_refuses_the_connection_is_an_error() {
        let path = std::env::temp_dir().join(format!("aither-refusing-bus-{}", std::process::id()));
        // A socket file nothing listens on refuses connections.
        drop(std::os::unix::net::UnixListener::bind(&path).expect("a free socket path"));
        let address = format!("unix:path={}", path.display());
        let builder = Builder::address(address.as_str());
        let result = Watcher::connect_with(builder).await;
        std::fs::remove_file(&path).expect("the socket file exists");
        assert!(matches!(result, Err(ConnectivityError::Register { .. })));
    }
}
