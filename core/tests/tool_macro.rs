//! Integration tests for `#[tool]` inside a crate that depends directly on
//! `aither-core` (without `aither` in scope), covering `mut` bindings and
//! parameter-level doc, `serde`, and `schemars` attributes.
#![cfg(feature = "derive")]
#![allow(clippy::unused_async)]

use aither_core::llm::tool::{ToolContext, ToolDefinition, Tools};
use aither_core::tool;
use schemars::JsonSchema;
use serde::Deserialize;

const fn default_limit() -> usize {
    5
}

/// Searches documents with mutable bindings and field attributes.
#[tool]
async fn search_docs(
    /// Search query string (including CJK like 中文, 日本語, 한국어).
    #[schemars(length(min = 1))]
    mut query: String,
    mut cx: ToolContext,
    /// Maximum results to return.
    #[serde(default = "default_limit", rename = "maxResults")]
    mut limit: usize,
) -> Result<String, core::convert::Infallible> {
    let _ = &mut cx;
    query = query.trim().to_owned();
    limit = limit.max(1);
    Ok(format!("{query}:{limit}"))
}

/// Normalizes a single annotated parameter.
#[tool]
async fn normalize_tag(
    /// Tag identifier to normalize.
    #[serde(rename = "tagName")]
    mut tag: String,
) -> Result<String, core::convert::Infallible> {
    tag.make_ascii_lowercase();
    Ok(tag)
}

/// Payload for single-struct mutable parameter test.
#[derive(JsonSchema, Deserialize)]
struct MutStructArgs {
    count: u32,
}

#[tool]
async fn bump_struct(mut args: MutStructArgs) -> Result<String, core::convert::Infallible> {
    args.count += 1;
    Ok(args.count.to_string())
}

#[test]
fn preserves_parameter_docs_serde_renames_and_schemars_constraints() {
    let schema = ToolDefinition::new(&SearchDocs).arguments_openai_schema();
    assert_eq!(schema["type"], "object");
    let props = schema["properties"].as_object().expect("properties");
    assert_eq!(
        props["query"]["description"],
        "Search query string (including CJK like 中文, 日本語, 한국어)."
    );
    assert!(
        props.contains_key("maxResults") && !props.contains_key("limit"),
        "serde rename must apply to generated struct field: {schema}"
    );
    assert_eq!(
        props["maxResults"]["description"],
        "Maximum results to return."
    );

    let single_schema = ToolDefinition::new(&NormalizeTag).arguments_openai_schema();
    assert_eq!(
        single_schema["properties"]["tagName"]["description"],
        "Tag identifier to normalize."
    );
}

#[tokio::test]
async fn executes_tools_against_aither_core_with_mut_bindings_and_defaults() {
    let mut tools = Tools::new();
    tools.register(SearchDocs).expect("registers SearchDocs");
    tools
        .register(NormalizeTag)
        .expect("registers NormalizeTag");
    tools.register(BumpStruct).expect("registers BumpStruct");

    let res = tools
        .call(
            "search_docs",
            r#"{"query": "  中文 日本語 한국어  "}"#,
            ToolContext::new(),
        )
        .await
        .expect("default limit applied");
    assert_eq!(res.as_text(), Some("中文 日本語 한국어:5"));

    let res = tools
        .call(
            "search_docs",
            r#"{"query": "rust", "maxResults": 12}"#,
            ToolContext::new(),
        )
        .await
        .expect("explicit maxResults applied");
    assert_eq!(res.as_text(), Some("rust:12"));

    let res = tools
        .call(
            "normalize_tag",
            r#"{"tagName": "AITHER"}"#,
            ToolContext::new(),
        )
        .await
        .expect("single annotated arg executes");
    assert_eq!(res.as_text(), Some("aither"));

    let res = tools
        .call("bump_struct", r#"{"count": 9}"#, ToolContext::new())
        .await
        .expect("mut struct arg executes");
    assert_eq!(res.as_text(), Some("10"));
}
