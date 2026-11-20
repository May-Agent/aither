//! Smoke test for the ACP client against a real agent.
//!
//! Spawns `devin acp`, initializes, opens a session in `/tmp`, sets the
//! `model` config option to `swe-2-max`, then prompts the agent to reply
//! with a single word, printing streamed text and the stop reason.
//!
//! Requires an authenticated `devin` CLI on `PATH`.
//!
//! ```console
//! cargo run -p aither-acp --example devin_smoke --features process
//! ```

use std::path::Path;
use std::sync::{Arc, Mutex};

use aither_acp::client::{
    Client, ClientError, MessageKind, NotificationHandler, PermissionHandler, SessionEvent,
};
use aither_acp::domain::ids::ConfigId;
use aither_acp::domain::{
    ConfigValue, ConfigValueById, Implementation, PermissionOutcome, PermissionRequest,
    SelectedOutcome,
};
use aither_acp::rpc::Notification;
use aither_acp::vendor::devin;
use futures_lite::StreamExt;

/// Approves the first offered option of every permission request.
struct Allow;

impl PermissionHandler for Allow {
    fn request(
        &self,
        request: PermissionRequest,
    ) -> impl std::future::Future<Output = PermissionOutcome> + Send {
        eprintln!("[permission] {}", request.title);
        std::future::ready(PermissionOutcome::Selected(SelectedOutcome {
            option_id: request
                .options
                .first()
                .map_or_else(|| "deny".into(), |option| option.option_id.clone()),
            meta: None,
        }))
    }
}

/// Prints devin's private notifications and accumulates agent text.
#[derive(Clone, Default)]
struct Printer {
    /// Accumulated agent message text.
    text: Arc<Mutex<String>>,
}

impl NotificationHandler for Printer {
    fn notify(&self, notification: Notification) -> impl std::future::Future<Output = ()> + Send {
        match devin::notification(&notification) {
            Some(devin::DevinNotification::Output(output)) => {
                eprintln!("[devin:{}] {}", output.channel, output.message);
            }
            Some(devin::DevinNotification::McpServersChanged(_)) => {
                eprintln!("[devin] MCP servers changed");
            }
            Some(devin::DevinNotification::Other { method, .. }) => {
                eprintln!("[devin] {method}");
            }
            None => eprintln!("[notification] {}", notification.method),
        }
        std::future::ready(())
    }
}

#[tokio::main]
async fn main() -> Result<(), ClientError> {
    let printer = Printer::default();
    let (client, connection) = Client::builder(Implementation::new("devin-smoke", "0"))
        .permissions(Allow)
        .notification(printer.clone())
        .spawn("devin", &["acp".to_string()], &[], Path::new("/tmp"))
        .await?;
    let connection = tokio::spawn(connection);

    eprintln!(
        "[initialize] agent={} protocol={}",
        client.agent_info().map_or("?", |i| i.name.as_str()),
        client.protocol_version(),
    );

    let session = client.new_session("/tmp").open().await?;
    eprintln!("[session] {}", session.id());

    session
        .set_config(
            ConfigId::from_string("model"),
            ConfigValue::Id(ConfigValueById {
                value: "swe-2-max".into(),
            }),
        )
        .await?;

    let printer2 = printer.clone();
    let mut events = session.subscribe();
    let printer_task = tokio::spawn(async move {
        while let Some(event) = events.next().await {
            if let SessionEvent::MessageDelta {
                kind: MessageKind::Agent,
                text: Some(text),
                ..
            } = &event
            {
                print!("{text}");
                printer2.text.lock().expect("text").push_str(text);
            }
        }
    });

    eprintln!("[prompt] streaming response:");
    let outcome = session
        .prompt_run("Reply with the single word pong.")
        .await?;
    eprintln!("\n[done] outcome={outcome:?}");
    eprintln!("[done] full text: {}", printer.text.lock().unwrap());

    printer_task.abort();
    connection.abort();
    Ok(())
}
