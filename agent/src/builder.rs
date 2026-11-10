//! Builder for constructing agents with custom configuration.
//!
//! The builder pattern allows fluent configuration of agents with
//! tools, hooks, and various settings.

use std::marker::PhantomData;
#[cfg(feature = "skills")]
use std::sync::Arc;

use aither_core::{
    LanguageModel,
    llm::{Tool, model::Parameters, tool::ToolDefinition},
};
#[cfg(feature = "skills")]
use aither_skills::SkillRegistry;

use crate::{
    agent::Agent,
    config::AgentConfig,
    context::{
        AnyContextSource, AnyEventSource, AnyFileReferences, AnyLiveSource, AnyStateSource,
        Context, ContextPolicy, ContextSource, EventSource, ExternalContent, ExternalSlot,
        FileReferences, HandoffDocument, HandoffSchema, InstructionSlot, LiveSource, StateSource,
        SystemLayout,
    },
    hook::{HCons, Hook},
    request_parameters::WithParameters,
    tools::AgentTools,
};

#[cfg(feature = "mcp")]
use aither_mcp::McpConnection;

/// Builder for constructing agents with custom configuration.
///
/// # Type Parameters
///
/// - `M`: the conversation model
/// - `H`: the composed hooks
/// - `C`: the model compaction runs on, when [`compaction_model`](Self::compaction_model)
///   sets one
/// - `Schema`: the handoff schema, see [`handoff`](Self::handoff)
///
/// # Example
///
/// ```rust,ignore
/// let agent = Agent::builder(claude)
///     .request_parameters(parameters)
///     .instruction("You are a helpful assistant.")
///     .tool(FileSystemTool::read_only("."))
///     .build();
/// ```
#[must_use]
pub struct AgentBuilder<M, H = (), C = M, Schema = HandoffDocument> {
    model: M,
    /// The model compaction runs on instead of the session's own, with the
    /// parameters of its requests.
    compaction_model: Option<WithParameters<C>>,
    hooks: H,
    parameters: Parameters,
    config: AgentConfig,
    context: Context,
    prompt_cache: crate::context::PromptCache,
    clock: crate::AnyClock,
    context_sources: Vec<AnyContextSource>,
    live_sources: Vec<AnyLiveSource>,
    state_sources: Vec<AnyStateSource>,
    event_sources: Vec<AnyEventSource>,
    stop_hook: Option<crate::AnyStopHook>,
    fork_source: Option<crate::fork::ForkSource>,
    identity: aither_core::llm::AgentIdentity,
    tools: AgentTools,
    connectivity: crate::connectivity::Watch,
    restore_dir: Option<std::path::PathBuf>,
    file_references: Option<AnyFileReferences>,
    /// The window state a fork inherits, which replaces initialization.
    inherited: Option<Inherited>,
    #[cfg(feature = "skills")]
    skill_registry: Option<Arc<SkillRegistry>>,
    /// The handoff schema compactions ask for; no value is held.
    handoff_schema: PhantomData<fn() -> Schema>,
}

/// What a fork carries over from the main agent besides its context.
#[derive(Clone)]
struct Inherited {
    budget: crate::context::ContextBudget,
    profile: aither_core::llm::model::Profile,
    effort: crate::effort::EffortAnchor,
    definitions: Vec<ToolDefinition>,
}

impl<M, H, C, Schema> std::fmt::Debug for AgentBuilder<M, H, C, Schema> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentBuilder")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl<M: LanguageModel> AgentBuilder<M> {
    /// Creates a new agent builder with default configuration.
    pub fn new(model: M) -> Self {
        Self {
            model,
            compaction_model: None,
            hooks: (),
            parameters: Parameters::default(),
            config: AgentConfig::default(),
            context: Context::default(),
            prompt_cache: crate::context::PromptCache::new(uuid::Uuid::new_v4().to_string()),
            clock: crate::AnyClock::default(),
            context_sources: Vec::new(),
            live_sources: Vec::new(),
            state_sources: Vec::new(),
            event_sources: Vec::new(),
            stop_hook: None,
            fork_source: None,
            identity: aither_core::llm::AgentIdentity::Main,
            tools: AgentTools::new(),
            connectivity: crate::connectivity::Watch::default(),
            restore_dir: None,
            file_references: None,
            inherited: None,
            #[cfg(feature = "skills")]
            skill_registry: None,
            handoff_schema: PhantomData,
        }
    }
}

impl<M, H, C, Schema> AgentBuilder<M, H, C, Schema>
where
    M: LanguageModel + 'static,
    C: LanguageModel + 'static,
    H: Hook,
    Schema: HandoffSchema,
{
    /// Sets the identity every tool call of the built agent carries as the
    /// agent making it. An agent is the main agent unless an embedder builds
    /// it as a subagent.
    pub fn identity(mut self, identity: aither_core::llm::AgentIdentity) -> Self {
        self.identity = identity;
        self
    }

    /// Publishes every request the built agent sends to `source`, which is
    /// where a fork starts from.
    pub fn fork_source(mut self, source: crate::fork::ForkSource) -> Self {
        self.fork_source = Some(source);
        self
    }

    /// Sets the maximum number of iterations (turns).
    ///
    /// The agent will stop and return an error if this limit is exceeded.
    pub const fn max_iterations(mut self, limit: usize) -> Self {
        self.config.max_iterations = limit;
        self
    }

    /// Sets the context management policy: when compaction starts, where the
    /// window pauses, and the budgets compaction and recovery work within.
    pub const fn context_policy(mut self, policy: ContextPolicy) -> Self {
        self.config.context_policy = policy;
        self
    }

    /// Sets the full agent configuration.
    pub const fn config(mut self, config: AgentConfig) -> Self {
        self.config = config;
        self
    }

    /// Registers a source rendered into the tail of every request.
    ///
    /// Sources are rendered in registration order, and each is told when the
    /// request it contributed to was sent.
    pub fn context_source<S: ContextSource>(mut self, source: S) -> Self {
        self.context_sources.push(AnyContextSource::new(source));
        self
    }

    /// Registers live content: a source rendered into the cacheable prefix,
    /// after the system prompt and before the handoff, for content too large
    /// to resend uncached in every request's tail.
    ///
    /// Blocks render in registration order. A change after a request carried
    /// the block reaches the model as the diff alert the source writes for
    /// it, and the new block replaces the old one at the next reassembly or
    /// compaction (see [`LiveSource`]).
    ///
    /// # Panics
    ///
    /// Panics if a live source with the same [`TAG`](LiveSource::TAG) is
    /// already registered.
    pub fn live_source<S: LiveSource>(mut self, source: S) -> Self {
        assert!(
            self.live_sources
                .iter()
                .all(|registered| registered.tag() != S::TAG),
            "live source `{}` is registered twice",
            S::TAG
        );
        self.live_sources.push(AnyLiveSource::new(source));
        self
    }

    /// Registers state: keyed conditions the model is told about, as a
    /// request is assembled, only when a value differs from the one it last
    /// saw (see [`StateSource`]).
    pub fn state_source<S: StateSource>(mut self, source: S) -> Self {
        self.state_sources.push(AnyStateSource::new(source));
        self
    }

    /// Registers events, each told to the model once with the next request
    /// (see [`EventSource`]).
    pub fn event_source<S: EventSource>(mut self, source: S) -> Self {
        self.event_sources.push(AnyEventSource::new(source));
        self
    }

    /// Decides through `hook` whether the run ends each time the model ends
    /// its turn. Without one, the run ends unless a context source objects.
    pub fn stop_hook(mut self, hook: crate::AnyStopHook) -> Self {
        self.stop_hook = Some(hook);
        self
    }

    /// Sets the skill registry used for the skill catalog compaction offers.
    #[cfg(feature = "skills")]
    pub fn skill_registry(mut self, registry: Arc<SkillRegistry>) -> Self {
        self.skill_registry = Some(registry);
        self
    }

    /// Appends an instruction to the system prompt, verbatim. Instructions
    /// render in the order they are added; aither supplies none of its own.
    pub fn instruction(mut self, content: impl Into<String>) -> Self {
        let instructions = self.context.instructions_mut();
        let slot = InstructionSlot::owned(format!("instruction-{}", instructions.len()));
        instructions.place(slot.clone());
        instructions.insert(&slot, content);
        self
    }

    /// Appends an instruction to the system prompt wrapped in a `tag` tag.
    pub fn tagged_instruction(mut self, tag: &'static str, content: &str) -> Self {
        let instructions = self.context.instructions_mut();
        let slot = InstructionSlot::new(tag);
        instructions.place(slot.clone());
        instructions.insert_tagged(&slot, content);
        self
    }

    /// Appends the external-content block `name`: memory, a repository's
    /// instructions, a fetched page or a file, rendered after every
    /// instruction inside a boundary it cannot close.
    pub fn external_content(mut self, name: &'static str, content: ExternalContent) -> Self {
        let external = self.context.external_mut();
        let slot = ExternalSlot::new(name);
        external.place(slot.clone());
        external.insert(&slot, content);
        self
    }

    /// Lays the system prompt out in `layout`'s order, for a prompt whose
    /// blocks are placed by name instead of by the order they are added in.
    ///
    /// # Panics
    ///
    /// Panics if a block already added has no slot in `layout`.
    pub fn system_layout(mut self, layout: SystemLayout) -> Self {
        self.context.set_layout(layout);
        self
    }

    /// Sets the parameters of every request aither sends the conversation
    /// model: the agent's turns, its compactions on that model, its cache
    /// keep-alives and the first request of a fork.
    ///
    /// aither sets the fields it owns per request over them — the prompt-
    /// cache controls, structured output on a compaction, and the token cap
    /// of a keep-alive — and sends every other field as given here. The
    /// [`compaction_model`](Self::compaction_model) takes its own.
    pub fn request_parameters(mut self, parameters: Parameters) -> Self {
        self.parameters = parameters;
        self
    }

    /// Overrides the session's prompt-cache controller.
    ///
    /// The default derives a fresh session key per agent, so every rebuild
    /// asks for a new cache shard; a host that rebuilds its agent supplies
    /// the session's identity (and the replaced agent's epoch, through
    /// [`PromptCache::with_epoch`](crate::context::PromptCache::with_epoch))
    /// so the key stays the same across rebuilds.
    pub fn prompt_cache(mut self, cache: crate::context::PromptCache) -> Self {
        self.prompt_cache = cache;
        self
    }

    /// Sets the Claude cache retention of the session's prompt cache,
    /// keeping its session key; the cache lifetime follows the retention.
    pub fn claude_cache_ttl(mut self, ttl: aither_core::llm::model::ClaudePromptCacheTtl) -> Self {
        self.prompt_cache = self.prompt_cache.with_ttl(ttl);
        self
    }

    /// Sets the conversation's cache lifetime, keeping the rest of its
    /// prompt cache. Idle compaction and idle reassembly follow it.
    pub fn cache_lifetime(mut self, lifetime: crate::context::CacheLifetime) -> Self {
        self.prompt_cache = self.prompt_cache.with_lifetime(lifetime);
        self
    }

    /// Sets the time source idle compaction, idle reassembly and cache
    /// warmth read; the system clock by default.
    pub fn clock(mut self, clock: impl crate::Clock) -> Self {
        self.clock = crate::AnyClock::new(clock);
        self
    }

    /// Sets the directory files are restored from after a compaction.
    pub fn restore_dir(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.restore_dir = Some(path.into());
        self
    }

    /// Sets how a compaction's file index finds the files tool results name.
    pub fn file_references(mut self, references: impl FileReferences) -> Self {
        self.file_references = Some(AnyFileReferences::new(references));
        self
    }

    /// Sets the handoff schema every compaction asks the model for, in place
    /// of [`HandoffDocument`].
    ///
    /// The schema's type is the structured output the model fills; its
    /// [`HandoffSchema`] implementation gives the instruction, the retry,
    /// the validation and the text the conversation reads after the handoff.
    pub fn handoff<S2: HandoffSchema>(self) -> AgentBuilder<M, H, C, S2> {
        AgentBuilder {
            model: self.model,
            compaction_model: self.compaction_model,
            hooks: self.hooks,
            parameters: self.parameters,
            config: self.config,
            context: self.context,
            prompt_cache: self.prompt_cache,
            clock: self.clock,
            context_sources: self.context_sources,
            live_sources: self.live_sources,
            state_sources: self.state_sources,
            event_sources: self.event_sources,
            stop_hook: self.stop_hook,
            fork_source: self.fork_source,
            identity: self.identity,
            tools: self.tools,
            connectivity: self.connectivity,
            restore_dir: self.restore_dir,
            file_references: self.file_references,
            inherited: self.inherited,
            #[cfg(feature = "skills")]
            skill_registry: self.skill_registry,
            handoff_schema: PhantomData,
        }
    }

    /// Sets the model compaction runs on, instead of the session's own.
    ///
    /// Without one, compaction reads the conversation's prompt cache on the
    /// session's model, and an idle session is compacted shortly before the
    /// cache expires ([`Agent::compact_when_idle`]). A different model cannot
    /// read that cache, so with one set compaction requests go to it, idle
    /// compaction never runs, and an idle gap past the cache lifetime
    /// reassembles the window instead.
    ///
    /// Compaction requests to it carry `parameters`, the model's own, in
    /// place of the session model's, under the cache controls and structured
    /// output aither sets on every compaction.
    pub fn compaction_model<C2: LanguageModel>(
        self,
        model: C2,
        parameters: Parameters,
    ) -> AgentBuilder<M, H, C2, Schema> {
        AgentBuilder {
            model: self.model,
            compaction_model: Some(WithParameters {
                model,
                parameters: Box::new(parameters),
            }),
            hooks: self.hooks,
            parameters: self.parameters,
            config: self.config,
            context: self.context,
            prompt_cache: self.prompt_cache,
            clock: self.clock,
            context_sources: self.context_sources,
            live_sources: self.live_sources,
            state_sources: self.state_sources,
            event_sources: self.event_sources,
            stop_hook: self.stop_hook,
            fork_source: self.fork_source,
            identity: self.identity,
            tools: self.tools,
            connectivity: self.connectivity,
            restore_dir: self.restore_dir,
            file_references: self.file_references,
            inherited: self.inherited,
            #[cfg(feature = "skills")]
            skill_registry: self.skill_registry,
            handoff_schema: PhantomData,
        }
    }

    /// Adds a hook to intercept agent operations.
    ///
    /// Hooks are composed using the `HCons` pattern, allowing multiple
    /// hooks to be chained at compile time.
    pub fn hook<NH: Hook>(self, hook: NH) -> AgentBuilder<M, HCons<NH, H>, C, Schema> {
        AgentBuilder {
            model: self.model,
            compaction_model: self.compaction_model,
            hooks: HCons::new(hook, self.hooks),
            parameters: self.parameters,
            config: self.config,
            context: self.context,
            prompt_cache: self.prompt_cache,
            clock: self.clock,
            context_sources: self.context_sources,
            live_sources: self.live_sources,
            state_sources: self.state_sources,
            event_sources: self.event_sources,
            stop_hook: self.stop_hook,
            fork_source: self.fork_source,
            identity: self.identity,
            tools: self.tools,
            connectivity: self.connectivity,
            restore_dir: self.restore_dir,
            file_references: self.file_references,
            inherited: self.inherited,
            #[cfg(feature = "skills")]
            skill_registry: self.skill_registry,
            handoff_schema: PhantomData,
        }
    }

    /// Registers a tool.
    ///
    /// # Panics
    ///
    /// Panics if the tool cannot be registered: another tool already uses
    /// its name, or it carries no description for the model to read. Both
    /// are mistakes in the calling program.
    pub fn tool<T: Tool + 'static>(mut self, tool: T) -> Self {
        self.tools
            .register(tool)
            .expect("tool registration must succeed: duplicate name or missing description");
        self
    }

    /// Makes the built agent a fork of the main agent that sent `fork`.
    ///
    /// The agent starts from the main agent's request: its window, with the
    /// live tail as the last message, its request parameters and its
    /// prompt-cache identity, so its first request is the main agent's
    /// followed by the prompt it is given and reads the main agent's cache.
    /// Nothing is initialized on top of that, since initialization would
    /// rewrite the window.
    ///
    /// The tools registered on this builder must present exactly
    /// [`ForkSnapshot::definitions`](crate::ForkSnapshot::definitions), or
    /// the first request would diverge from the cached prefix at the tool
    /// definitions.
    pub fn forked_from(mut self, fork: &crate::fork::ForkSnapshot) -> Self {
        self.context = fork.context.clone();
        self.prompt_cache = fork.prompt_cache.clone();
        self.parameters = fork.request_parameters.clone();
        self.config.context_policy = fork.context_policy;
        self.inherited = Some(Inherited {
            budget: fork.budget,
            profile: fork.profile.clone(),
            effort: fork.effort,
            definitions: fork.definitions.clone(),
        });
        self
    }

    /// Registers an MCP connection.
    ///
    /// All tools from the MCP server will be available for the agent to use.
    /// You can call this method multiple times to register multiple MCP servers.
    #[cfg(feature = "mcp")]
    pub fn mcp(mut self, conn: McpConnection) -> Self {
        self.tools.register_mcp(conn);
        self
    }

    /// Ends a provider response stream in flight when the machine wakes from
    /// sleep or loses its network.
    ///
    /// The attempt fails with an [`AgentError::Llm`](crate::AgentError::Llm)
    /// whose failure is
    /// [`FailureKind::Network`](aither_core::llm::FailureKind::Network), so the
    /// caller retries instead of sitting on a dead socket. Nothing about the
    /// change reaches the agent's context.
    #[cfg(feature = "connectivity")]
    pub fn connectivity(mut self, connectivity: aither_connectivity::Connectivity) -> Self {
        self.connectivity = crate::connectivity::Watch::new(connectivity);
        self
    }

    /// Returns the definitions of the tools registered on this builder so far.
    ///
    /// These are the definitions the built agent sends to the model, in the
    /// order the tool registry holds them.
    #[must_use]
    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        self.tools.active_definitions()
    }

    /// Builds the agent.
    ///
    /// # Panics
    ///
    /// Panics when the agent is a fork and its tools do not present the
    /// definitions the main agent's request carried.
    pub fn build(self) -> Agent<M, H, C, Schema> {
        let inherited = self.inherited.as_ref();
        if let Some(inherited) = inherited {
            assert_eq!(
                self.tools.active_definitions(),
                inherited.definitions,
                "a fork's tools must present the definitions of the request it starts from"
            );
        }
        // A fork starts initialized, from what the main agent carried over.
        let initialized = inherited.is_some();
        let (budget, effort, profile) = inherited.map_or_else(Default::default, |inherited| {
            (
                inherited.budget,
                inherited.effort,
                Some(inherited.profile.clone()),
            )
        });
        Agent {
            warmth: crate::CacheWarmth::new(self.prompt_cache.lifetime(), self.clock),
            model: std::sync::Arc::new(self.model),
            compaction_model: self.compaction_model.map(|compactor| WithParameters {
                model: std::sync::Arc::new(compactor.model),
                parameters: compactor.parameters,
            }),
            parameters: Box::new(self.parameters),
            effort,
            idle_compacted: false,
            tools: self.tools,
            hooks: self.hooks,
            config: self.config,
            context: self.context,
            budget,
            prompt_cache: self.prompt_cache,
            context_sources: self.context_sources,
            live_sources: self.live_sources,
            state_sources: self.state_sources,
            event_sources: self.event_sources,
            reassembled: false,
            prefix_rewrites: crate::context::PrefixRewrites::default(),
            requests_sent: 0,
            compaction: crate::compaction_state::CompactionState::default(),
            compaction_failure: None,
            compaction_announced: false,
            tool_results: std::collections::VecDeque::new(),
            profile,
            compaction_profile: None,
            initialized,
            connectivity: self.connectivity,
            stop_hook: self.stop_hook,
            restore_dir: self.restore_dir,
            file_references: self.file_references,
            last_request: None,
            delivering: None,
            fork_source: self.fork_source,
            identity: self.identity,
            cache_stats: crate::CacheStats::new(),
            #[cfg(feature = "skills")]
            skill_registry: self.skill_registry,
            handoff_schema: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow;

    use schemars::JsonSchema;
    use serde::Deserialize;

    use futures_core::Stream;

    // Mock error type for testing
    #[derive(Debug)]
    struct MockError;

    impl std::fmt::Display for MockError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "mock error")
        }
    }

    impl std::error::Error for MockError {}

    impl aither_core::llm::Classify for MockError {
        fn failure(&self) -> aither_core::llm::Failure {
            aither_core::llm::Failure::permanent()
        }
    }

    // Mock LLM for testing
    #[derive(Clone)]
    struct MockLlm;

    impl LanguageModel for MockLlm {
        type Error = MockError;

        fn respond(
            &self,
            _request: aither_core::llm::LLMRequest,
        ) -> impl Stream<Item = Result<aither_core::llm::Event, Self::Error>> + Send {
            futures_lite::stream::empty()
        }

        fn profile(
            &self,
        ) -> impl std::future::Future<
            Output = Result<aither_core::llm::model::Profile, Self::Error>,
        > + Send {
            std::future::ready(Ok(aither_core::llm::model::Profile::new(
                "mock",
                "test",
                "mock-model",
                "A mock model for testing",
                100_000,
            )))
        }
    }

    // Mock tool
    struct MockTool;

    /// Does nothing, for tests that only care that a tool was registered.
    #[derive(Debug, JsonSchema, Deserialize)]
    struct MockArgs;

    impl Tool for MockTool {
        fn name(&self) -> Cow<'static, str> {
            "mock_tool".into()
        }

        type Arguments = MockArgs;
        type Res = aither_core::llm::ToolResult;

        fn call(
            &self,
            _args: Self::Arguments,
            _cx: aither_core::llm::ToolContext,
        ) -> impl std::future::Future<Output = aither_core::Result<Self::Res>> + Send {
            std::future::ready(Ok(aither_core::llm::ToolResult::text("ok")))
        }
    }

    // Mock hook
    struct MockHook;

    impl Hook for MockHook {}

    #[test]
    fn test_builder_basic() {
        let agent = AgentBuilder::new(MockLlm).build();
        assert_eq!(agent.tools.definitions(), []);
    }

    #[test]
    fn test_builder_with_tool() {
        let agent = AgentBuilder::new(MockLlm).tool(MockTool).build();
        assert_eq!(agent.tools.definitions().len(), 1);
    }

    #[test]
    fn instructions_render_in_the_order_they_were_added() {
        let agent = AgentBuilder::new(MockLlm)
            .instruction("first")
            .tagged_instruction("second", "two")
            .instruction("third")
            .build();
        let rendered: Vec<String> = agent
            .context
            .build_request(&[])
            .iter()
            .map(|message| message.content().to_string())
            .collect();
        let joined = rendered.join("\n");
        let first = joined.find("first").expect("first instruction renders");
        let second = joined.find("two").expect("second instruction renders");
        let third = joined.find("third").expect("third instruction renders");
        assert!(first < second && second < third);
    }

    #[test]
    fn the_agent_holds_no_instruction_the_embedder_did_not_add() {
        let agent = AgentBuilder::new(MockLlm).build();
        assert!(agent.context.instructions().is_empty());
        assert!(agent.context.external().is_empty());
    }

    #[test]
    fn test_builder_with_hook() {
        let _agent = AgentBuilder::new(MockLlm).hook(MockHook).build();
    }

    #[test]
    fn test_builder_with_multiple_hooks() {
        let _agent = AgentBuilder::new(MockLlm)
            .hook(MockHook)
            .hook(MockHook)
            .build();
    }

    /// Options set before a step that changes the builder's type survive it.
    #[test]
    fn options_survive_the_steps_that_change_the_builders_type() {
        let policy = ContextPolicy {
            handoff_reserve: 1_234,
            ..ContextPolicy::default()
        };
        let agent = AgentBuilder::new(MockLlm)
            .context_policy(policy)
            .max_iterations(7)
            .hook(MockHook)
            .compaction_model(MockLlm, Parameters::default())
            .build();
        assert_eq!(agent.config.context_policy, policy);
        assert_eq!(agent.config.max_iterations, 7);
    }

    #[test]
    fn test_builder_default_config() {
        let agent = AgentBuilder::new(MockLlm).build();
        assert_eq!(
            agent.config.max_iterations,
            AgentConfig::default().max_iterations
        );
    }
}
