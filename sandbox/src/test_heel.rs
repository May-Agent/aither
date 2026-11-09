//! The `heel` binary the repository's distribution pins, for tests.
//!
//! Tests take heel the way an embedder does: from the pinned entries of
//! `patch/third-party-tools.json`, downloaded, digest-verified and cached
//! once, never from a hand-set `HEEL_BIN`.

#![allow(clippy::missing_panics_doc, clippy::must_use_candidate)]

use std::path::PathBuf;

use heel::HeelBinary;
use serde_json::Value;

use crate::{HostProgram, ToolDistribution, ToolRuntimeError, ToolTarget};

fn pinned_manifest() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../patch/third-party-tools.json");
    serde_json::from_slice(&std::fs::read(path).expect("the repository pins its tools"))
        .expect("the pinned tools are JSON")
}

/// The pinned `heel` entries of every platform, as manifest `host_programs`.
pub fn host_programs() -> Vec<Value> {
    pinned_manifest()["host_programs"]
        .as_array()
        .expect("the pinned tools list host programs")
        .iter()
        .filter(|program| program["tool"] == "heel")
        .cloned()
        .collect()
}

/// A distribution holding only the pinned `heel`.
pub fn distribution() -> ToolDistribution {
    let distribution = ToolDistribution::from_json(
        &serde_json::to_vec(&serde_json::json!({
            "artifacts": [],
            "host_programs": host_programs(),
        }))
        .expect("the manifest serializes"),
    )
    .expect("the pinned heel is a valid manifest");
    // CI hands its job token down so release downloads are authenticated.
    match std::env::var("GITHUB_TOKEN") {
        Ok(token) => distribution.with_github_token(token),
        Err(_) => distribution,
    }
}

/// The pinned `heel`, prepared into the build directory's cache and verified
/// against the linked library.
///
/// # Errors
/// Fails when the distribution cannot provide heel for this platform or the
/// binary is not the linked library's version.
pub async fn pinned() -> Result<HeelBinary, ToolRuntimeError> {
    let target_dir = std::env::var_os("CARGO_TARGET_DIR").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target"),
        PathBuf::from,
    );
    let heel = distribution()
        .prepare_host_program(
            HostProgram::Heel,
            &ToolTarget::native()?,
            &target_dir.join("aither-test-tools"),
        )
        .await?;
    let path = heel.path().to_path_buf();
    Ok(blocking::unblock(move || HeelBinary::verify(path)).await?)
}
