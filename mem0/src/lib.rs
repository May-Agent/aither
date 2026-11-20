//! Conversation memory for aither agents.
//!
//! Extracts durable facts from a conversation, reconciles them against what is
//! already stored, and exposes the result both as a query API and as tools an
//! agent can call.

use std::str::FromStr;
use std::sync::Arc;

use aither_core::embedding::EmbeddingModel;
use aither_core::llm::ToolContext;
use aither_core::llm::{LLMRequest, LanguageModel, Message, Role, Tool, ToolResult};
use anyhow::Context;
use askama::Template;
use async_lock::{Mutex, RwLock};
use llm::{Action, ExtractedFacts, MemoryDecision};
use store::{MemoryStore, SearchFilters};
use tracing::debug;
use uuid::Uuid;

pub mod error;
/// Structured types the language model produces during fact extraction.
pub mod llm;
/// Storage backends for extracted memories.
pub mod store;

pub use error::{Mem0Error, Result};
pub use store::{InMemoryStore, Memory, SearchResult};

struct ConversationLine<'a> {
    role: Role,
    content: &'a str,
}

#[derive(Template)]
#[template(path = "extract_facts_user.txt", escape = "none")]
struct ExtractFactsPrompt<'a> {
    messages: &'a [ConversationLine<'a>],
}

#[derive(Template)]
#[template(path = "decide_operation_user.txt", escape = "none")]
struct DecideOperationPrompt<'a> {
    fact: &'a str,
    existing_memories: &'a [SearchResult],
}

#[derive(Template)]
#[template(path = "relevant_memories.txt", escape = "none")]
struct RelevantMemoriesTemplate<'a> {
    results: &'a [SearchResult],
}

/// Tool that lets an agent search its own memory.
#[derive(Debug)]
pub struct SearchTool<L, E, S> {
    inner: Mem0<L, E, S>,
}

impl<L, E, S> Tool for SearchTool<L, E, S>
where
    L: LanguageModel + 'static,
    E: EmbeddingModel + 'static,
    S: MemoryStore + 'static,
{
    fn description(&self) -> std::borrow::Cow<'static, str> {
        "Search previously stored memories for anything relevant to a query. \
         Pass the query as a plain string. Returns matching memories as text, \
         or nothing if none are relevant."
            .into()
    }

    type Arguments = String;
    type Res = ToolResult;
    fn name(&self) -> std::borrow::Cow<'static, str> {
        "search_memories".into()
    }

    async fn call(
        &self,
        arguments: Self::Arguments,
        _cx: ToolContext,
    ) -> aither_core::Result<Self::Res> {
        let result = self
            .inner
            .retrieve_formatted(&arguments, 50)
            .await
            .context("Failed to retrieve memory")?;
        Ok(ToolResult::text(result))
    }
}

/// Tool that lets an agent record facts worth remembering.
#[derive(Debug)]
pub struct AddFactTool<L, E, S> {
    inner: Mem0<L, E, S>,
}

impl<L, E, S> Tool for AddFactTool<L, E, S>
where
    L: LanguageModel + 'static,
    E: EmbeddingModel + 'static,
    S: MemoryStore + 'static,
{
    fn description(&self) -> std::borrow::Cow<'static, str> {
        "Record facts worth remembering across conversations. Pass a list of \
         short, self-contained statements. Existing memories are updated or \
         removed automatically when a new fact supersedes them."
            .into()
    }

    type Arguments = Vec<String>;
    type Res = ToolResult;
    fn name(&self) -> std::borrow::Cow<'static, str> {
        "add_fact".into()
    }

    async fn call(
        &self,
        arguments: Self::Arguments,
        _cx: ToolContext,
    ) -> aither_core::Result<Self::Res> {
        self.inner
            .add_fact(arguments)
            .await
            .context("Fail to add fact")?;
        Ok(ToolResult::Done)
    }
}

/// Configuration for Mem0.
#[derive(Debug, Clone)]
pub struct Config {
    /// Number of similar memories to retrieve for update context.
    pub retrieve_count: usize,
    /// User ID to associate with memories.
    pub user_id: Option<String>,
    /// Agent ID to associate with memories.
    pub agent_id: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            retrieve_count: 5,
            user_id: None,
            agent_id: None,
        }
    }
}

struct Mem0Inner<L, E, S> {
    llm: L,
    embedder: E,
    store: RwLock<S>,
    /// Serializes fact extraction and reconciliation so concurrent `add` /
    /// `add_fact` calls reconcile against each other's updates in order.
    write_gate: Mutex<()>,
    config: Config,
}

/// A conversation memory.
///
/// Cloning shares the same underlying store and reconciliation state.
pub struct Mem0<L, E, S> {
    inner: Arc<Mem0Inner<L, E, S>>,
}

impl<L, E, S> std::fmt::Debug for Mem0<L, E, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mem0").finish_non_exhaustive()
    }
}

impl<L, E, S> Clone for Mem0<L, E, S> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<L, E, S> Mem0<L, E, S>
where
    L: LanguageModel,
    E: EmbeddingModel,
    S: MemoryStore,
{
    /// Create a new memory over the given model, embedder and store.
    #[must_use]
    pub fn new(llm: L, embedder: E, store: S, config: Config) -> Self {
        Self {
            inner: Arc::new(Mem0Inner {
                llm,
                embedder,
                store: RwLock::new(store),
                write_gate: Mutex::new(()),
                config,
            }),
        }
    }

    /// Add a new interaction to memory.
    ///
    /// This triggers the extraction and update pipeline:
    /// 1. Extract facts from the messages.
    /// 2. For each fact, retrieve similar memories.
    /// 3. Decide on an operation (Add, Update, Delete, Noop).
    /// 4. Execute the operation.
    ///
    /// # Errors
    ///
    /// Returns an error if fact extraction, embedding, or the store fails.
    pub async fn add(&self, messages: &[Message]) -> Result<()> {
        let _write = self.inner.write_gate.lock().await;
        let facts = extract_facts(&self.inner.llm, messages).await?;
        self.inner.reconcile_facts(facts).await
    }

    /// Add new facts to memory.
    ///
    /// Reconciles one batch of facts at a time so concurrent callers observe
    /// each other's updates without racing.
    ///
    /// # Errors
    ///
    /// Returns an error if the model call, embedding, or the store fails.
    pub async fn add_fact(&self, facts: Vec<String>) -> Result<()> {
        let _write = self.inner.write_gate.lock().await;
        self.inner.reconcile_facts(facts).await
    }

    /// Search for relevant memories.
    ///
    /// # Errors
    ///
    /// Returns an error if the query cannot be embedded or the store fails.
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<store::SearchResult>> {
        let embedding = self
            .inner
            .embedder
            .embed(query)
            .await
            .map_err(Mem0Error::Embedding)?;
        let filters = self.inner.filters();
        let store = self.inner.store.read().await;
        store.search(&embedding, limit, filters).await
    }

    /// A tool the agent can call to record facts into this memory.
    #[must_use]
    pub fn add_fact_tool(&self) -> AddFactTool<L, E, S> {
        AddFactTool {
            inner: self.clone(),
        }
    }

    /// A tool the agent can call to search this memory.
    #[must_use]
    pub fn search_tool(&self) -> SearchTool<L, E, S> {
        SearchTool {
            inner: self.clone(),
        }
    }

    /// Return all stored memories.
    ///
    /// # Errors
    ///
    /// Returns an error if the store cannot be read.
    pub async fn memories(&self) -> Result<Vec<Memory>> {
        let store = self.inner.store.read().await;
        store.all().await
    }

    /// Retrieve relevant memories and format them for context injection.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying search fails.
    pub async fn retrieve_formatted(&self, query: &str, limit: usize) -> Result<String> {
        let results = self.search(query, limit).await?;
        if results.is_empty() {
            return Ok(String::new());
        }

        RelevantMemoriesTemplate { results: &results }
            .render()
            .map_err(|err| Mem0Error::Extraction(err.to_string()))
    }
}

impl<L, E, S> Mem0Inner<L, E, S>
where
    L: LanguageModel,
    E: EmbeddingModel,
    S: MemoryStore,
{
    fn filters(&self) -> SearchFilters {
        SearchFilters {
            user_id: self.config.user_id.clone(),
            agent_id: self.config.agent_id.clone(),
        }
    }

    async fn reconcile_facts(&self, facts: Vec<String>) -> Result<()> {
        for fact in facts {
            let embedding = self
                .embedder
                .embed(&fact)
                .await
                .map_err(Mem0Error::Embedding)?;

            debug!("Embedding generated for fact: {}", fact);

            let existing_memories = {
                let store = self.store.read().await;
                store
                    .search(&embedding, self.config.retrieve_count, self.filters())
                    .await?
            };

            debug!(
                "Found {} similar existing memories for fact.",
                existing_memories.len()
            );

            let decision = decide_operation(&self.llm, &fact, &existing_memories).await?;

            match decision.action {
                Action::Add => {
                    let mut memory = Memory::new(fact, embedding);
                    if let Some(uid) = &self.config.user_id {
                        memory = memory.with_user_id(uid);
                    }
                    if let Some(aid) = &self.config.agent_id {
                        memory = memory.with_agent_id(aid);
                    }
                    self.store.write().await.add(memory).await?;
                }
                Action::Update => {
                    if let (Some(id_str), Some(content)) =
                        (decision.memory_id, decision.new_content)
                        && let Ok(id) = Uuid::from_str(&id_str)
                    {
                        let new_embedding = self
                            .embedder
                            .embed(&content)
                            .await
                            .map_err(Mem0Error::Embedding)?;

                        let mut store = self.store.write().await;
                        if let Some(mut existing) = store.get(id).await? {
                            existing.content = content;
                            existing.embedding = new_embedding;
                            existing.updated_at = time::OffsetDateTime::now_utc();
                            store.update(existing).await?;
                        }
                        drop(store);
                    }
                }
                Action::Delete => {
                    if let Some(id_str) = decision.memory_id
                        && let Ok(id) = Uuid::from_str(&id_str)
                    {
                        self.store.write().await.delete(id).await?;
                    }
                }
                Action::Noop => {}
            }
        }
        Ok(())
    }
}

async fn extract_facts<L: LanguageModel>(llm: &L, messages: &[Message]) -> Result<Vec<String>> {
    let lines: Vec<ConversationLine<'_>> = messages
        .iter()
        .map(|m| ConversationLine {
            role: m.role(),
            content: m.content(),
        })
        .collect();
    let user_prompt = ExtractFactsPrompt { messages: &lines }
        .render()
        .map_err(|err| Mem0Error::Extraction(err.to_string()))?;

    let system_prompt = include_str!("../prompts/extractor.txt");
    let request = LLMRequest::new(vec![
        Message::system(system_prompt),
        Message::user(user_prompt),
    ]);

    let extracted: ExtractedFacts = llm
        .generate(request)
        .await
        .map_err(|err| Mem0Error::Llm(anyhow::Error::new(err)))?;
    Ok(extracted.facts)
}

async fn decide_operation<L: LanguageModel>(
    llm: &L,
    fact: &str,
    existing_memories: &[store::SearchResult],
) -> Result<MemoryDecision> {
    let user_prompt = DecideOperationPrompt {
        fact,
        existing_memories,
    }
    .render()
    .map_err(|err| Mem0Error::Extraction(err.to_string()))?;

    let system_prompt = include_str!("../prompts/manager.txt");
    let request = LLMRequest::new(vec![
        Message::system(system_prompt),
        Message::user(user_prompt),
    ]);

    debug!("Deciding operation for fact: {}", fact);

    llm.generate(request)
        .await
        .map_err(|err| Mem0Error::Llm(anyhow::Error::new(err)))
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::future::Future;
    use std::sync::Mutex as StdMutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use aither_core::llm::{Classify, Event, Failure, model::Profile};
    use futures_lite::Stream;

    use super::*;

    #[derive(Debug)]
    struct MockLlmError;

    impl std::fmt::Display for MockLlmError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("mock llm failure")
        }
    }

    impl std::error::Error for MockLlmError {}

    impl Classify for MockLlmError {
        fn failure(&self) -> Failure {
            Failure::permanent()
        }
    }

    #[derive(Clone)]
    struct ScriptedLlm {
        responses: Arc<StdMutex<VecDeque<std::result::Result<String, MockLlmError>>>>,
        prompts: Arc<StdMutex<Vec<String>>>,
    }

    impl ScriptedLlm {
        fn new(
            responses: impl IntoIterator<Item = std::result::Result<String, MockLlmError>>,
        ) -> Self {
            Self {
                responses: Arc::new(StdMutex::new(responses.into_iter().collect())),
                prompts: Arc::new(StdMutex::new(Vec::new())),
            }
        }
    }

    impl LanguageModel for ScriptedLlm {
        type Error = MockLlmError;

        fn respond(
            &self,
            request: LLMRequest,
        ) -> impl Stream<Item = std::result::Result<Event, Self::Error>> + Send {
            if let Some(last) = request.messages().last() {
                self.prompts
                    .lock()
                    .expect("prompts lock")
                    .push(last.content().to_string());
            }
            let next = self
                .responses
                .lock()
                .expect("responses lock")
                .pop_front()
                .expect("scripted response available");
            futures_lite::stream::iter(vec![next.map(Event::Text)])
        }

        fn profile(
            &self,
        ) -> impl Future<Output = std::result::Result<Profile, Self::Error>> + Send {
            std::future::ready(Ok(Profile::new("mock", "test", "mock", "Mock", 32_000)))
        }
    }

    #[derive(Clone)]
    struct CountingEmbedder {
        fail_on_call: Option<usize>,
        calls: Arc<AtomicUsize>,
    }

    impl CountingEmbedder {
        fn ok() -> Self {
            Self {
                fail_on_call: None,
                calls: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn failing_on(call_index: usize) -> Self {
            Self {
                fail_on_call: Some(call_index),
                calls: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    impl EmbeddingModel for CountingEmbedder {
        fn dim(&self) -> usize {
            3
        }

        fn embed(&self, text: &str) -> impl Future<Output = aither_core::Result<Vec<f32>>> + Send {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            let result = if self.fail_on_call == Some(call) {
                Err(anyhow::Error::msg("embedder backend offline"))
            } else {
                let len = u16::try_from(text.len()).map_or_else(|_| f32::from(u16::MAX), f32::from);
                Ok(vec![1.0, len + 1.0, 0.5])
            };
            std::future::ready(result)
        }
    }

    #[tokio::test]
    async fn add_update_delete_and_formatted_retrieval_use_templates() {
        // CJK test coverage (Chinese, Japanese, Korean)
        let first_fact = "用户喜欢茶 / ユーザーはお茶が好き / 사용자는 차를 좋아함";
        let updated_fact = "用户喜欢抹茶 / ユーザーは抹茶が好き / 사용자는 말차를 좋아함";

        let llm = ScriptedLlm::new([
            Ok(format!(r#"{{"facts":["{first_fact}"]}}"#)),
            Ok(r#"{"action":"ADD","memory_id":null,"new_content":null,"reasoning":"new"}"#.into()),
        ]);
        let prompts = Arc::clone(&llm.prompts);
        let mem0 = Mem0::new(
            llm,
            CountingEmbedder::ok(),
            InMemoryStore::new(),
            Config {
                retrieve_count: 5,
                user_id: Some("u1".into()),
                agent_id: Some("a1".into()),
            },
        );

        mem0.add(&[Message::user(first_fact)]).await.unwrap();
        let stored = mem0.memories().await.unwrap();
        assert_eq!(stored.len(), 1);
        let id = stored[0].id;
        assert_eq!(stored[0].content, first_fact);
        assert_eq!(stored[0].user_id.as_deref(), Some("u1"));

        let captured = prompts.lock().expect("prompts").clone();
        assert!(
            captured[0].starts_with("Extract facts from the following conversation:\n\nUser: ")
        );
        assert!(captured[1].starts_with("New Fact: "));
        assert!(captured[1].ends_with("Decide the operation."));

        let formatted = mem0.retrieve_formatted("tea", 5).await.unwrap();
        assert_eq!(formatted, format!("Relevant Memories:\n- {first_fact}"));

        // Update the existing memory via `InMemoryStore` and verify `Delete`.
        let llm_update_delete = ScriptedLlm::new([
            Ok(format!(
                r#"{{"action":"UPDATE","memory_id":"{id}","new_content":"{updated_fact}","reasoning":"refined"}}"#
            )),
            Ok(format!(
                r#"{{"action":"DELETE","memory_id":"{id}","new_content":null,"reasoning":"retracted"}}"#
            )),
        ]);
        let mut seeded_store = InMemoryStore::new();
        seeded_store.add(stored[0].clone()).await.unwrap();
        let mem0_ops = Mem0::new(
            llm_update_delete,
            CountingEmbedder::ok(),
            seeded_store,
            Config {
                retrieve_count: 5,
                user_id: Some("u1".into()),
                agent_id: Some("a1".into()),
            },
        );

        mem0_ops
            .add_fact(vec![updated_fact.to_string()])
            .await
            .unwrap();
        let after_update = mem0_ops.memories().await.unwrap();
        assert_eq!(after_update.len(), 1);
        assert_eq!(after_update[0].content, updated_fact);

        mem0_ops
            .add_fact(vec!["retracted".to_string()])
            .await
            .unwrap();
        assert!(mem0_ops.memories().await.unwrap().is_empty());
        assert_eq!(mem0_ops.retrieve_formatted("tea", 5).await.unwrap(), "");
    }

    #[tokio::test]
    async fn embedding_failures_report_embedding_variant_across_add_update_and_search() {
        // 1. Search embed failure -> Mem0Error::Embedding
        let mem_search = Mem0::new(
            ScriptedLlm::new([]),
            CountingEmbedder::failing_on(0),
            InMemoryStore::new(),
            Config::default(),
        );
        let search_err = mem_search.search("query", 5).await.unwrap_err();
        assert!(
            matches!(search_err, Mem0Error::Embedding(_)),
            "expected Mem0Error::Embedding on search, got {search_err:?}"
        );

        // 2. Add embed failure -> Mem0Error::Embedding
        let mem_add = Mem0::new(
            ScriptedLlm::new([]),
            CountingEmbedder::failing_on(0),
            InMemoryStore::new(),
            Config::default(),
        );
        let add_err = mem_add
            .add_fact(vec!["fact".to_string()])
            .await
            .unwrap_err();
        assert!(
            matches!(add_err, Mem0Error::Embedding(_)),
            "expected Mem0Error::Embedding on add, got {add_err:?}"
        );

        // 3. Update re-embed failure (second embed call) -> Mem0Error::Embedding
        let mut store = InMemoryStore::new();
        let existing = Memory::new("initial", vec![1.0, 2.0, 0.5]);
        let id = existing.id;
        store.add(existing).await.unwrap();

        let mem_update = Mem0::new(
            ScriptedLlm::new([Ok(format!(
                r#"{{"action":"UPDATE","memory_id":"{id}","new_content":"updated","reasoning":"change"}}"#
            ))]),
            CountingEmbedder::failing_on(1),
            store,
            Config::default(),
        );
        let update_err = mem_update
            .add_fact(vec!["updated".to_string()])
            .await
            .unwrap_err();
        assert!(
            matches!(update_err, Mem0Error::Embedding(_)),
            "expected Mem0Error::Embedding on update re-embed, got {update_err:?}"
        );
    }

    #[tokio::test]
    async fn llm_and_store_errors_surface_as_typed_variants() {
        let mem_llm = Mem0::new(
            ScriptedLlm::new([Err(MockLlmError)]),
            CountingEmbedder::ok(),
            InMemoryStore::new(),
            Config::default(),
        );
        let llm_err = mem_llm.add(&[Message::user("hello")]).await.unwrap_err();
        assert!(matches!(llm_err, Mem0Error::Llm(_)), "{llm_err:?}");

        let mut store = InMemoryStore::new();
        let missing = Memory::new("orphan", vec![1.0, 0.0, 0.0]);
        let store_err = store.update(missing).await.unwrap_err();
        assert!(matches!(store_err, Mem0Error::Store(_)), "{store_err:?}");
        assert_eq!(
            store
                .search(&vec![1.0, 0.0, 0.0], 0, SearchFilters::default())
                .await
                .unwrap()
                .len(),
            0
        );
    }
}
