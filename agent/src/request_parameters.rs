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
