//! Brave Search API provider.
//!
//! [Brave Search](https://brave.com/search/api/) offers a privacy-first search API
//! built on an independent web index.
//!
//! # Example
//!
//! ```no_run
//! use aither_websearch::{BraveSearch, SearchProvider};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let provider = BraveSearch::new("YOUR_API_KEY");
//! let results = provider.search("rust programming", 5).await?;
//! for result in results {
//!     println!("{}: {}", result.title, result.url);
//! }
//! # Ok(())
//! # }
//! ```

use crate::{SearchError, SearchProvider, SearchResult};
use serde::Deserialize;
use zenwave::{Client, client, header};

/// Brave Search API endpoint.
const BRAVE_API_URL: &str = "https://api.search.brave.com/res/v1/web/search";

/// Brave Search provider.
///
/// Uses the [Brave Search API](https://brave.com/search/api/) to perform web searches.
/// Requires an API key which can be obtained from the Brave Search API dashboard.
#[derive(Debug, Clone)]
pub struct BraveSearch {
    api_key: String,
}

impl BraveSearch {
    /// Create a new Brave Search provider with the given API key.
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
        }
    }
}

impl SearchProvider for BraveSearch {
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>, SearchError> {
        let url = format!("{}?q={}&count={}", BRAVE_API_URL, urlencoded(query), limit);

        let mut backend = client();
        let response: BraveResponse = backend
            .get(&url)
            .map_err(SearchError::from)?
            .header("X-Subscription-Token", &self.api_key)
            .map_err(SearchError::from)?
            .header(header::ACCEPT.as_str(), "application/json")
            .map_err(SearchError::from)?
            .header(header::USER_AGENT.as_str(), "aither-websearch/0.1")
            .map_err(SearchError::from)?
            .json()
            .await
            .map_err(SearchError::from)?;

        Ok(response.web.map_or_default(|web| {
            web.results
                .into_iter()
                .map(|r| SearchResult {
                    title: r.title,
                    url: r.url,
                    snippet: r.description.unwrap_or_default(),
                })
                .collect()
        }))
    }
}

#[derive(Debug, Deserialize)]
struct BraveResponse {
    web: Option<BraveWebResults>,
}

#[derive(Debug, Deserialize)]
struct BraveWebResults {
    results: Vec<BraveResult>,
}

#[derive(Debug, Deserialize)]
struct BraveResult {
    title: String,
    url: String,
    description: Option<String>,
}

fn urlencoded(s: &str) -> String {
    url::form_urlencoded::byte_serialize(s.as_bytes()).collect()
}
