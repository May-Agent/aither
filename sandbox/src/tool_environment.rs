use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::ContainerPath;
use crate::shell_session::{ShellBackend, ShellSessionRegistry};
use crate::spawn::Backend;
use crate::ssh::{PreparedSsh, SshEnvironment};
use crate::tool_distribution::ToolDistribution;
use crate::tool_runtime::{
    HostProgram, PreparedEnvironment, PreparedTools, ToolCatalog, ToolRuntimeError, ToolTarget,
};
use crate::tool_transport::{
    ContainerToolTransport, LocalToolTransport, SshToolTransport, ToolTransport,
    ensure_command_line_tools, probe_tool_environment, successful_output,
};

/// The home backend's tools, typed by where that backend sees them.
#[derive(Debug, Clone)]
enum PreparedHome {
    Local(PreparedTools<PathBuf>),
    Container {
        container_id: String,
        tools: PreparedTools<ContainerPath>,
    },
}

impl PreparedHome {
    const fn backend(&self) -> ShellBackend {
        match self {
            Self::Local(_) => ShellBackend::Local,
            Self::Container { .. } => ShellBackend::Container,
        }
    }

    const fn catalog(&self) -> &ToolCatalog {
        match self {
            Self::Local(tools) => tools.catalog(),
            Self::Container { tools, .. } => tools.catalog(),
        }
    }
}

/// What preparing an SSH server on its first command draws on: the
/// distribution the home environment was prepared from and its cache.
#[derive(Debug)]
struct ToolSource {
    distribution: ToolDistribution,
    cache_root: PathBuf,
}

#[derive(Debug, Clone)]
pub struct PreparedToolEnvironments {
    home: PreparedHome,
    catalog: ToolCatalog,
    /// `None` only for registries tests prepare by hand.
    source: Option<Arc<ToolSource>>,
}

impl ShellSessionRegistry {
    /// Prepares the home environment before an agent can use this registry.
    ///
    /// SSH servers are not prepared here: each is prepared on its first
    /// command, from the same distribution and cache.
    ///
    /// # Errors
    /// Fails on unsupported platforms, missing artifacts, integrity failures
    /// and a `home` that is no home environment.
    pub async fn initialize_tools(
        mut self,
        home: &Backend,
        distribution: &ToolDistribution,
        cache_root: &Path,
    ) -> Result<Self, ToolRuntimeError> {
        let home = if let Backend::Container { runtime, .. } = home {
            let transport = ContainerToolTransport::new(runtime.clone());
            let info = probe_tool_environment(&transport).await?;
            let tools = distribution
                .image_tools(&info.target)?
                .verify(&transport)
                .await?;
            PreparedHome::Container {
                container_id: runtime.container_id().into(),
                tools,
            }
        } else if matches!(home, Backend::Host { .. }) {
            let target = ToolTarget::native()?;
            ensure_command_line_tools(&LocalToolTransport, &target).await?;
            let artifacts = cache_root.join("artifacts");
            let (bundle, heel) = futures::try_join!(
                distribution.load(&target, &artifacts),
                distribution.prepare_host_program(HostProgram::Heel, &target, cache_root),
            )?;
            PreparedHome::Local(
                bundle
                    .prepare_local(&cache_root.join("runtimes"))
                    .await?
                    .with_heel(heel),
            )
        } else {
            return Err(ToolRuntimeError::NotInitialized(
                "an SSH server is no home environment".into(),
            ));
        };
        let catalog = home.catalog().clone();
        self.prepared_tools = Some(PreparedToolEnvironments {
            home,
            catalog,
            source: Some(Arc::new(ToolSource {
                distribution: distribution.clone(),
                cache_root: cache_root.to_path_buf(),
            })),
        });
        Ok(self)
    }

    /// Prepared environments, once the registry was prepared.
    fn prepared_environments(&self) -> Result<&PreparedToolEnvironments, ToolRuntimeError> {
        self.prepared_tools
            .as_ref()
            .ok_or_else(|| ToolRuntimeError::NotInitialized("execution environments".into()))
    }

    /// Checks that the prepared tools are exactly the ones `home` runs
    /// with: preparation is bound to the home backend and the container it
    /// mounted into, so a registry prepared for another home describes an
    /// environment nobody prepared.
    ///
    /// # Errors
    /// Rejects an uninitialized registry, one prepared for another home
    /// backend and one prepared in another container.
    pub(crate) fn check_home(&self, home: &Backend) -> Result<(), ToolRuntimeError> {
        let prepared = self.prepared_environments()?;
        let backend = match home {
            Backend::Host { .. } => ShellBackend::Local,
            Backend::Container { .. } => ShellBackend::Container,
            Backend::Ssh { .. } => {
                return Err(ToolRuntimeError::EnvironmentChanged(
                    PreparedEnvironment::HomeBackend,
                ));
            }
        };
        if prepared.home.backend() != backend {
            return Err(ToolRuntimeError::EnvironmentChanged(
                PreparedEnvironment::HomeBackend,
            ));
        }
        if let (PreparedHome::Container { container_id, .. }, Backend::Container { runtime, .. }) =
            (&prepared.home, home)
            && runtime.container_id() != container_id.as_str()
        {
            return Err(ToolRuntimeError::EnvironmentChanged(
                PreparedEnvironment::Container,
            ));
        }
        Ok(())
    }

    /// The prepared tools of a local home backend, as the host sees them.
    ///
    /// Programs the host itself runs, such as external hooks, take their
    /// baseline tools from this directory, so they run exactly the verified
    /// commands the terminal's own commands do.
    ///
    /// # Errors
    /// Rejects an uninitialized registry, one whose configuration changed
    /// after preparation, and one whose home backend is not local, whose
    /// tools are not runnable on the host; use [`Self::host_tools`] for tools
    /// the host runs.
    pub fn local_tools(&self) -> Result<&PreparedTools<PathBuf>, ToolRuntimeError> {
        match &self.prepared_environments()?.home {
            PreparedHome::Local(tools) => Ok(tools),
            PreparedHome::Container { .. } => {
                Err(ToolRuntimeError::NotInitialized("local environment".into()))
            }
        }
    }

    /// Baseline tools runnable on the host, for programs the host itself runs.
    ///
    /// A local home backend's prepared tools are returned as they are. When
    /// the home backend is a container, its tools target another platform,
    /// so the same distribution prepares the host target into the same cache
    /// with the same pinning and verification.
    ///
    /// # Errors
    /// Rejects an uninitialized registry and one whose configuration changed
    /// after preparation, and fails when the distribution has no verified
    /// artifacts for the host platform.
    pub async fn host_tools(
        &self,
        distribution: &ToolDistribution,
        cache_root: &Path,
    ) -> Result<PreparedTools<PathBuf>, ToolRuntimeError> {
        if let PreparedHome::Local(tools) = &self.prepared_environments()?.home {
            return Ok(tools.clone());
        }
        distribution
            .load(&ToolTarget::native()?, &cache_root.join("artifacts"))
            .await?
            .prepare_local(&cache_root.join("runtimes"))
            .await
    }

    pub(crate) fn container_tools(
        &self,
    ) -> Result<&PreparedTools<ContainerPath>, ToolRuntimeError> {
        match &self.prepared_environments()?.home {
            PreparedHome::Container { tools, .. } => Ok(tools),
            PreparedHome::Local(_) => Err(ToolRuntimeError::NotInitialized(
                "container environment".into(),
            )),
        }
    }

    /// The server's tools and `heel`, prepared the first time a command
    /// needs them: concurrent first commands wait for one preparation, and a
    /// failed one is tried again by the next command.
    pub(crate) async fn ssh_tools<'e>(
        &self,
        environment: &'e SshEnvironment,
    ) -> Result<&'e PreparedSsh, ToolRuntimeError> {
        let prepared = self.prepared_environments()?;
        let catalog = prepared.catalog.clone();
        let source = prepared.source.clone();
        let client = self.ssh_environments().client().clone();
        environment
            .prepared(|| async move {
                let source = source.ok_or_else(|| {
                    ToolRuntimeError::NotInitialized("the tool distribution".into())
                })?;
                let transport = SshToolTransport::over(client, environment.server().clone());
                let info = probe_tool_environment(&transport).await?;
                ensure_command_line_tools(&transport, &info.target).await?;
                let bundle = source
                    .distribution
                    .load(&info.target, &source.cache_root.join("artifacts"))
                    .await?;
                let (tools, heel) = futures::try_join!(
                    bundle.prepare_remote(&transport, &info.cache_root),
                    remote_heel(&transport, &info.target, &source),
                )?;
                if tools.catalog() != &catalog {
                    return Err(ToolRuntimeError::CatalogMismatch);
                }
                Ok(PreparedSsh { heel, tools })
            })
            .await
    }

    /// Catalog derived only from environments that completed initialization.
    ///
    /// # Errors
    /// Rejects an uninitialized registry, or one whose home backend or
    /// container changed after preparation.
    pub fn prepared_catalog(&self) -> Result<&ToolCatalog, ToolRuntimeError> {
        self.prepared_environments()
            .map(|prepared| &prepared.catalog)
    }

    #[cfg(test)]
    pub(crate) fn with_test_tools(self, tools: PreparedTools<PathBuf>) -> Self {
        self.with_test_home(PreparedHome::Local(tools))
    }

    #[cfg(test)]
    pub(crate) fn with_test_container_tools(
        self,
        tools: PreparedTools<ContainerPath>,
        container_id: String,
    ) -> Self {
        self.with_test_home(PreparedHome::Container {
            container_id,
            tools,
        })
    }

    #[cfg(test)]
    fn with_test_home(mut self, home: PreparedHome) -> Self {
        let catalog = home.catalog().clone();
        self.prepared_tools = Some(PreparedToolEnvironments {
            home,
            catalog,
            source: None,
        });
        self
    }

    /// Adds `server` already prepared with `tools` and `heel`, as its first
    /// command would leave it; its approval is still asked.
    #[cfg(test)]
    pub(crate) fn with_test_ssh(
        self,
        server: crate::SshServer,
        heel: Option<String>,
        tools: PreparedTools<PathBuf>,
    ) -> Self {
        let name = server.name.clone();
        self.add_ssh_server(server).expect("a test server is valid");
        self.ssh_environments()
            .get(&name)
            .expect("the server was just added")
            .set_prepared(PreparedSsh { heel, tools });
        self
    }
}

/// The server's `heel`: the one it has, or the distribution's pinned `heel`
/// for the server's own target, installed there.
///
/// # Errors
/// Fails when the server has no `heel` and the distribution pins none for its
/// target.
async fn remote_heel(
    transport: &SshToolTransport,
    target: &ToolTarget,
    source: &ToolSource,
) -> Result<Option<String>, ToolRuntimeError> {
    let found = successful_output(
        transport
            .run(include_str!("../templates/ssh_heel_probe.sh"), None)
            .await?,
    )?;
    if let Some(path) = found.lines().next() {
        return Ok(Some(path.to_string()));
    }
    let local = source
        .distribution
        .prepare_host_program(HostProgram::Heel, target, &source.cache_root)
        .await?;
    let binary = async_fs::read(local.path()).await?;
    let installed = successful_output(
        transport
            .run(
                include_str!("../templates/ssh_heel_install.sh"),
                Some(binary),
            )
            .await?,
    )?;
    let path = installed
        .lines()
        .next()
        .ok_or(ToolRuntimeError::InvalidProbe)?;
    Ok(Some(path.to_string()))
}
