//! Approval of the network domains a sandbox's lists leave open.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::network_policy::{CompiledNetworkPolicy, NetworkVerdict};

/// One connection a sandboxed process makes to a domain the network lists
/// leave open.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DomainRequest {
    /// The host being connected to.
    pub host: String,
    /// The port being connected to.
    pub port: u16,
    /// The agent whose sandbox the process runs in, as the embedder labelled
    /// the sandbox; `None` for an unlabelled sandbox.
    pub agent: Option<String>,
}

/// Decides the connections the sandbox's network lists leave open.
///
/// The sandbox consults its [`CompiledNetworkPolicy`] first: a host the lists
/// allow or deny never reaches the approver. For the rest it asks
/// [`approve`](Self::approve) once per connection and enforces the answer for
/// that connection. Auditing, asking a user, a reviewer and remembering an
/// approval are the implementation's policy.
pub trait DomainApprover: Send + Sync {
    /// Whether the process may connect to `request`'s host and port.
    fn approve(&self, request: &DomainRequest) -> impl Future<Output = bool> + Send;
}

/// Approves every domain it is asked about.
#[derive(Debug, Clone, Copy, Default)]
pub struct ApproveAllDomains;

impl DomainApprover for ApproveAllDomains {
    fn approve(&self, _request: &DomainRequest) -> impl Future<Output = bool> + Send {
        std::future::ready(true)
    }
}

/// Denies every domain it is asked about, which is what a sandbox without an
/// approver does.
#[derive(Debug, Clone, Copy, Default)]
pub struct DenyAllDomains;

impl DomainApprover for DenyAllDomains {
    fn approve(&self, _request: &DomainRequest) -> impl Future<Output = bool> + Send {
        std::future::ready(false)
    }
}

/// Object-safe twin of [`DomainApprover`], used only through
/// [`AnyDomainApprover`].
trait DomainApproverImpl: Send + Sync {
    fn approve_boxed<'a>(
        &'a self,
        request: &'a DomainRequest,
    ) -> Pin<Box<dyn Future<Output = bool> + Send + 'a>>;
}

impl<T: DomainApprover> DomainApproverImpl for T {
    fn approve_boxed<'a>(
        &'a self,
        request: &'a DomainRequest,
    ) -> Pin<Box<dyn Future<Output = bool> + Send + 'a>> {
        Box::pin(self.approve(request))
    }
}

/// A cloneable handle on any [`DomainApprover`].
#[derive(Clone)]
pub struct AnyDomainApprover(Arc<dyn DomainApproverImpl>);

impl std::fmt::Debug for AnyDomainApprover {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnyDomainApprover").finish_non_exhaustive()
    }
}

impl AnyDomainApprover {
    /// Wraps a concrete approver.
    #[must_use]
    pub fn new(approver: impl DomainApprover + 'static) -> Self {
        Self(Arc::new(approver))
    }
}

impl DomainApprover for AnyDomainApprover {
    fn approve(&self, request: &DomainRequest) -> impl Future<Output = bool> + Send {
        let inner = Arc::clone(&self.0);
        let request = request.clone();
        async move { inner.approve_boxed(&request).await }
    }
}

/// The network policy a sandbox hands heel: the lists first, then the
/// approver for what they leave open, and a denial when there is none.
#[derive(Clone)]
pub struct DomainGate {
    pub(crate) lists: Option<CompiledNetworkPolicy>,
    pub(crate) approver: Option<AnyDomainApprover>,
    pub(crate) agent: Option<Arc<str>>,
}

impl DomainGate {
    pub(crate) async fn decide(&self, host: &str, port: u16) -> bool {
        let verdict = self
            .lists
            .as_ref()
            .map_or(NetworkVerdict::Open, |lists| lists.verdict(host));
        match verdict {
            NetworkVerdict::Allowed => true,
            NetworkVerdict::Denied => false,
            NetworkVerdict::Open => match &self.approver {
                Some(approver) => {
                    approver
                        .approve(&DomainRequest {
                            host: host.to_string(),
                            port,
                            agent: self.agent.as_deref().map(str::to_string),
                        })
                        .await
                }
                None => false,
            },
        }
    }
}

impl heel::NetworkPolicy for DomainGate {
    async fn check(&self, request: &heel::DomainRequest) -> bool {
        self.decide(request.host(), request.port()).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::{NetworkPolicyConfig, NetworkPolicyMode};

    /// Records what it is asked and answers a fixed verdict.
    struct Recording {
        answer: bool,
        asked: Arc<Mutex<Vec<DomainRequest>>>,
    }

    impl DomainApprover for Recording {
        fn approve(&self, request: &DomainRequest) -> impl Future<Output = bool> + Send {
            self.asked.lock().unwrap().push(request.clone());
            std::future::ready(self.answer)
        }
    }

    fn gate(
        mode: NetworkPolicyMode,
        hosts: &[&str],
        answer: Option<bool>,
    ) -> (DomainGate, Arc<Mutex<Vec<DomainRequest>>>) {
        let asked = Arc::new(Mutex::new(Vec::new()));
        let lists = NetworkPolicyConfig {
            mode,
            hosts: hosts.iter().map(|host| (*host).to_string()).collect(),
        }
        .compile()
        .unwrap();
        let approver = answer.map(|answer| {
            AnyDomainApprover::new(Recording {
                answer,
                asked: asked.clone(),
            })
        });
        (
            DomainGate {
                lists,
                approver,
                agent: Some(Arc::from("worker")),
            },
            asked,
        )
    }

    #[tokio::test]
    async fn a_domain_the_lists_leave_open_reaches_the_approver_with_host_port_and_agent() {
        let (gate, asked) = gate(NetworkPolicyMode::AllowList, &["*.example.com"], Some(true));
        assert!(gate.decide("registry.other.org", 8443).await);
        assert_eq!(
            *asked.lock().unwrap(),
            [DomainRequest {
                host: "registry.other.org".into(),
                port: 8443,
                agent: Some("worker".into()),
            }]
        );
    }

    #[tokio::test]
    async fn the_approvers_answer_is_enforced_for_the_connection() {
        let (refusing, asked) = gate(
            NetworkPolicyMode::AllowList,
            &["*.example.com"],
            Some(false),
        );
        assert!(!refusing.decide("registry.other.org", 443).await);
        assert_eq!(asked.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn a_domain_the_lists_decide_never_reaches_the_approver() {
        let (allow, asked) = gate(
            NetworkPolicyMode::AllowList,
            &["*.example.com"],
            Some(false),
        );
        assert!(allow.decide("api.example.com", 443).await);
        let (deny, denied_asked) =
            gate(NetworkPolicyMode::DenyList, &["*.tracker.net"], Some(true));
        assert!(!deny.decide("ads.tracker.net", 443).await);
        let (restricted, restricted_asked) = gate(NetworkPolicyMode::Restricted, &[], Some(true));
        assert!(!restricted.decide("example.com", 443).await);
        for asked in [asked, denied_asked, restricted_asked] {
            assert!(asked.lock().unwrap().is_empty());
        }
    }

    #[tokio::test]
    async fn without_lists_every_domain_is_open_and_without_an_approver_denied() {
        let (asking, asked) = gate(NetworkPolicyMode::Disabled, &[], Some(true));
        assert!(asking.decide("example.com", 80).await);
        assert_eq!(asked.lock().unwrap().len(), 1);
        let (none, _) = gate(NetworkPolicyMode::Disabled, &[], None);
        assert!(!none.decide("example.com", 80).await);
    }
}
