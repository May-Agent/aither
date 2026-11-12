//! The `heel` binary tests run under.
//!
//! `HEEL_BIN` names one when set. Otherwise tests take the `heel` the
//! repository pins in `sandbox/test-heel.json`: downloaded from its
//! pinned location once, verified against its pinned digest and cached under
//! the build directory.

#![allow(clippy::missing_panics_doc, clippy::must_use_candidate)]

use std::path::{Path, PathBuf};
use std::time::Duration;

use futures_lite::io::AsyncReadExt;
use heel::HeelBinary;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use zenwave::Client as _;

/// The pinned `heel` of one platform, as the manifest lists it.
#[derive(Debug, Deserialize)]
struct Pin {
    tool: String,
    target: PinTarget,
    url: String,
    size: u64,
    executable_sha256: String,
}

#[derive(Debug, Deserialize)]
struct PinTarget {
    os: String,
    arch: String,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    host_programs: Vec<Pin>,
}

/// Four attempts with 1 s, 2 s and 4 s between them and 60 s per attempt.
const ATTEMPTS: u32 = 4;
const ATTEMPT_TIMEOUT: Duration = Duration::from_secs(60);

/// The `heel` tests run under: `HEEL_BIN` when set, else the pinned release,
/// verified against the linked library.
///
/// # Errors
/// Fails when the pinned heel cannot be downloaded or fails its digest, or
/// the binary is not the linked library's version.
pub async fn binary() -> anyhow::Result<HeelBinary> {
    if std::env::var_os("HEEL_BIN").is_some() {
        return Ok(blocking::unblock(HeelBinary::locate).await?);
    }
    let path = pinned().await?;
    Ok(blocking::unblock(move || HeelBinary::verify(path)).await?)
}

fn target_dir() -> PathBuf {
    std::env::var_os("CARGO_TARGET_DIR").map_or_else(
        || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target"),
        PathBuf::from,
    )
}

fn pin() -> anyhow::Result<Pin> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("test-heel.json");
    let manifest: Manifest = serde_json::from_slice(&std::fs::read(path)?)?;
    let (os, arch) = (std::env::consts::OS, std::env::consts::ARCH);
    let mut pins = manifest
        .host_programs
        .into_iter()
        .filter(|pin| pin.tool == "heel" && pin.target.os == os && pin.target.arch == arch);
    let pin = pins
        .next()
        .ok_or_else(|| anyhow::anyhow!("no heel is pinned for {os} {arch}"))?;
    anyhow::ensure!(
        pins.next().is_none(),
        "heel is pinned twice for {os} {arch}"
    );
    Ok(pin)
}

/// The pinned heel in the build directory's cache, downloaded when absent.
async fn pinned() -> anyhow::Result<PathBuf> {
    let pin = pin()?;
    let name = if cfg!(windows) { "heel.exe" } else { "heel" };
    let directory = target_dir()
        .join("aither-test-tools/bin")
        .join(&pin.executable_sha256);
    let path = directory.join(name);
    if path.is_file() {
        return Ok(path);
    }
    let bytes = download(&pin).await?;
    anyhow::ensure!(
        hex::encode(Sha256::digest(&bytes)) == pin.executable_sha256,
        "the downloaded heel does not match its pinned digest"
    );
    blocking::unblock(move || store(&directory, &path, &bytes)).await
}

/// Whether `path` holds exactly `bytes`.
fn holds(path: &Path, bytes: &[u8]) -> bool {
    std::fs::read(path).is_ok_and(|have| have == bytes)
}

/// Writes `bytes` as an executable at `path` through a file of its own, so
/// processes preparing it at once never see a partial file.
///
/// A file already at `path` with these bytes is used as it is and never
/// replaced: another process may be running it, and Windows refuses to
/// replace an executable in use. A rename that loses the race to an identical
/// file is success.
fn store(directory: &Path, path: &Path, bytes: &[u8]) -> anyhow::Result<PathBuf> {
    std::fs::create_dir_all(directory)?;
    if holds(path, bytes) {
        return Ok(path.to_path_buf());
    }
    let partial = directory.join(format!("partial-{}", std::process::id()));
    std::fs::write(&partial, bytes)?;
    #[cfg(unix)]
    std::fs::set_permissions(
        &partial,
        std::os::unix::fs::PermissionsExt::from_mode(0o755),
    )?;
    let renamed = std::fs::rename(&partial, path);
    if let Err(error) = renamed {
        let identical = holds(path, bytes);
        std::fs::remove_file(&partial)?;
        anyhow::ensure!(
            identical,
            "cannot place the pinned heel at {}: {error}",
            path.display()
        );
    }
    Ok(path.to_path_buf())
}

/// Downloads the pinned release, retrying transport failures.
async fn download(pin: &Pin) -> anyhow::Result<Vec<u8>> {
    let mut backoff = Duration::from_secs(1);
    let mut attempt = 1;
    loop {
        match download_once(pin).await {
            Err(error) if attempt < ATTEMPTS => {
                tracing::warn!(%error, attempt, "heel download failed; retrying");
                async_io::Timer::after(backoff).await;
                backoff *= 2;
                attempt += 1;
            }
            result => return result,
        }
    }
}

async fn download_once(pin: &Pin) -> anyhow::Result<Vec<u8>> {
    futures_lite::future::or(
        async {
            let response = zenwave::client().get(pin.url.as_str())?.await?;
            anyhow::ensure!(
                response.status().is_success(),
                "{} answered {}",
                pin.url,
                response.status()
            );
            let mut bytes = Vec::new();
            response
                .into_body()
                .into_reader()
                .take(pin.size.saturating_add(1))
                .read_to_end(&mut bytes)
                .await?;
            anyhow::ensure!(
                bytes.len() as u64 == pin.size,
                "the downloaded heel is {} bytes, pinned at {}",
                bytes.len(),
                pin.size
            );
            Ok(bytes)
        },
        async {
            async_io::Timer::after(ATTEMPT_TIMEOUT).await;
            anyhow::bail!("downloading {} timed out", pin.url)
        },
    )
    .await
}
