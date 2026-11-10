//! The parameters a host gives the requests aither sends a model.
//!
//! A request aither builds starts from the parameters the host set for the
//! model, and aither then sets the fields it owns for that request:
//!
//! - [`cache`](Parameters::cache) on an agent's turns and compactions, from
//!   its [`PromptCache`](crate::context::PromptCache);
//! - [`structured_outputs`](Parameters::structured_outputs) on a structured
//!   request such as a compaction;
//! - [`max_tokens`](Parameters::max_tokens) and [`stream`](Parameters::stream)
//!   on a prompt-cache keep-alive.
//!
//! Every other field reaches the provider exactly as the host set it.

use aither_core::llm::model::Parameters;

/// A model and the parameters of the requests aither sends it, boxed: they
/// are set once and only cloned onto requests, and held inline they would
/// weigh on every future that holds an agent or its builder.
#[derive(Debug, Clone)]
pub struct WithParameters<M> {
    pub model: M,
    pub parameters: Box<Parameters>,
}

/// Checks on what a request carried to the provider.
#[cfg(test)]
pub mod test_support {
    use aither_core::llm::LLMRequest;
    use aither_core::llm::model::{Parameters, ReasoningEffort};

    /// Parameters a host gives a model, setting no field aither owns.
    pub fn host_parameters(effort: ReasoningEffort, temperature: f32) -> Parameters {
        Parameters::default()
            .reasoning_effort(effort)
            .include_reasoning(true)
            .temperature(temperature)
            .websearch(true)
    }

    /// `request`'s parameters without the cache controls and structured
    /// output aither sets per request: what is left of the host's.
    pub fn host_part(request: &LLMRequest) -> Parameters {
        let mut parameters = request.parameters().clone().without_cache();
        parameters.structured_outputs = false;
        parameters
    }

    /// Asserts `request` carries the prompt-cache controls of an agent's
    /// request: an `OpenAI` cache key and Claude breakpoints.
    pub fn assert_cache_controls(request: &LLMRequest) {
        let cache = &request.parameters().cache;
        assert!(
            cache
                .openai
                .as_ref()
                .is_some_and(|openai| openai.key.is_some()),
            "the request carries the prompt-cache key: {cache:?}"
        );
        assert!(
            cache.claude.is_some(),
            "the request carries the Claude cache breakpoints: {cache:?}"
        );
    }
}
