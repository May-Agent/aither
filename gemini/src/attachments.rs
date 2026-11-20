#[cfg(not(target_arch = "wasm32"))]
use crate::config::GeminiConfig;
#[cfg(not(target_arch = "wasm32"))]
use crate::error::GeminiError;
#[cfg(not(target_arch = "wasm32"))]
use crate::files::upload_file;
#[cfg(not(target_arch = "wasm32"))]
use aither_attachments::{FileCache, default_cache_dir};
#[cfg(not(target_arch = "wasm32"))]
use aither_core::llm::{Attachment, Message};
#[cfg(not(target_arch = "wasm32"))]
use std::{
    collections::HashMap,
    future::{Future, poll_fn},
    path::{Path, PathBuf},
    pin::Pin,
    task::Poll,
    time::SystemTime,
};
#[cfg(not(target_arch = "wasm32"))]
use url::Url;

#[cfg(not(target_arch = "wasm32"))]
const MAX_CONCURRENT_UPLOADS: usize = 4;

#[cfg(not(target_arch = "wasm32"))]
pub async fn resolve_messages(
    cfg: &GeminiConfig,
    messages: Vec<Message>,
) -> Result<Vec<Message>, GeminiError> {
    resolve_messages_with_cache_dir(cfg, messages, default_cache_dir()).await
}

#[cfg(not(target_arch = "wasm32"))]
async fn resolve_messages_with_cache_dir(
    cfg: &GeminiConfig,
    messages: Vec<Message>,
    cache_dir: impl AsRef<Path>,
) -> Result<Vec<Message>, GeminiError> {
    if messages.iter().all(|msg| msg.attachments().is_empty()) {
        return Ok(messages);
    }

    let mut cache = FileCache::open(cache_dir.as_ref().to_path_buf())
        .await
        .map_err(GeminiError::from)?;
    let mut cache_dirty = cache.prune_expired();

    let mut resolved = Vec::with_capacity(messages.len());
    for message in messages {
        if message.attachments().is_empty() {
            resolved.push(message);
            continue;
        }
        match message {
            Message::User {
                content,
                attachments,
            } => {
                let attachments =
                    resolve_attachment_list(cfg, &mut cache, attachments, &mut cache_dirty).await?;
                resolved.push(Message::User {
                    content,
                    attachments,
                });
            }
            Message::Tool {
                content,
                tool_call_id,
                attachments,
            } => {
                let attachments =
                    resolve_attachment_list(cfg, &mut cache, attachments, &mut cache_dirty).await?;
                resolved.push(Message::Tool {
                    content,
                    tool_call_id,
                    attachments,
                });
            }
            other => resolved.push(other),
        }
    }

    if cache_dirty {
        cache.save().await.map_err(GeminiError::from)?;
    }

    Ok(resolved)
}

#[cfg(not(target_arch = "wasm32"))]
struct UploadedGeminiFile {
    path: PathBuf,
    uri: String,
    expires_at: Option<SystemTime>,
    parsed_url: Url,
}

#[cfg(not(target_arch = "wasm32"))]
enum AttachmentPlan {
    Direct(Url),
    Local(PathBuf),
}

#[cfg(not(target_arch = "wasm32"))]
async fn resolve_attachment_list(
    cfg: &GeminiConfig,
    cache: &mut FileCache,
    attachments: Vec<Attachment>,
    cache_dirty: &mut bool,
) -> Result<Vec<Attachment>, GeminiError> {
    let provider = "gemini";
    let mut plans = Vec::with_capacity(attachments.len());
    let mut resolved_urls: HashMap<PathBuf, Url> = HashMap::new();
    let mut pending_uploads: Vec<PathBuf> = Vec::new();

    for attachment in &attachments {
        let url = attachment.url();
        match url.scheme() {
            "file" => {
                let path = url.to_file_path().map_err(|()| {
                    GeminiError::Api(
                        "Attachment file URL could not be converted to path".to_string(),
                    )
                })?;
                if !resolved_urls.contains_key(&path) && !pending_uploads.contains(&path) {
                    if let Some(entry) = cache
                        .get(&path, provider)
                        .await
                        .map_err(GeminiError::from)?
                    {
                        let cached_url = Url::parse(&entry.reference).map_err(|e| {
                            GeminiError::Api(format!("Invalid cached Gemini file URL: {e}"))
                        })?;
                        resolved_urls.insert(path.clone(), cached_url);
                    } else {
                        pending_uploads.push(path.clone());
                    }
                }
                plans.push(AttachmentPlan::Local(path));
            }
            "http" | "https" | "data" => plans.push(AttachmentPlan::Direct(url.clone())),
            other => {
                return Err(GeminiError::Api(format!(
                    "Unsupported attachment URL scheme: {other}"
                )));
            }
        }
    }

    if !pending_uploads.is_empty() {
        let uploaded =
            try_map_bounded(pending_uploads, MAX_CONCURRENT_UPLOADS, |path| async move {
                let file = upload_file(cfg, &path).await?;
                if !file.is_ready() {
                    return Err(GeminiError::Api(format!(
                        "Gemini file upload not ready (state: {:?})",
                        file.state
                    )));
                }
                if file.uri.is_empty() {
                    return Err(GeminiError::Api(
                        "Gemini file upload missing URI".to_string(),
                    ));
                }
                let parsed_url = Url::parse(&file.uri)
                    .map_err(|e| GeminiError::Api(format!("Invalid Gemini file URI: {e}")))?;
                let expires_at = file.expiration();
                Ok(UploadedGeminiFile {
                    path,
                    uri: file.uri,
                    expires_at,
                    parsed_url,
                })
            })
            .await?;

        for item in uploaded {
            cache
                .insert(&item.path, provider, item.uri, item.expires_at)
                .await
                .map_err(GeminiError::from)?;
            *cache_dirty = true;
            resolved_urls.insert(item.path, item.parsed_url);
        }
    }

    let mut next_attachments = Vec::with_capacity(attachments.len());
    for (attachment, plan) in attachments.into_iter().zip(plans) {
        let next_url = match plan {
            AttachmentPlan::Direct(url) => url,
            AttachmentPlan::Local(path) => resolved_urls
                .get(&path)
                .cloned()
                .expect("every local attachment was cached or uploaded"),
        };
        next_attachments.push(attachment.with_url(next_url));
    }
    Ok(next_attachments)
}

#[cfg(not(target_arch = "wasm32"))]
async fn try_map_bounded<T, U, E, F, Fut>(
    items: Vec<T>,
    limit: usize,
    mut f: F,
) -> Result<Vec<U>, E>
where
    F: FnMut(T) -> Fut,
    Fut: Future<Output = Result<U, E>>,
{
    let total = items.len();
    if total == 0 {
        return Ok(Vec::new());
    }
    let bound = limit.max(1).min(total);
    let mut iter = items.into_iter().enumerate();
    let mut active: Vec<(usize, Pin<Box<Fut>>)> = Vec::with_capacity(bound);
    let mut results: Vec<Option<U>> = (0..total).map(|_| None).collect();

    while active.len() < bound
        && let Some((idx, item)) = iter.next()
    {
        active.push((idx, Box::pin(f(item))));
    }

    while !active.is_empty() {
        let (done_pos, (idx, outcome)) = poll_fn(|cx| {
            for (pos, (item_idx, fut)) in active.iter_mut().enumerate() {
                if let Poll::Ready(res) = fut.as_mut().poll(cx) {
                    return Poll::Ready((pos, (*item_idx, res)));
                }
            }
            Poll::Pending
        })
        .await;

        let _ = active.swap_remove(done_pos);
        results[idx] = Some(outcome?);

        if let Some((next_idx, next_item)) = iter.next() {
            active.push((next_idx, Box::pin(f(next_item))));
        }
    }

    Ok(results
        .into_iter()
        .map(|slot| slot.expect("every input index completed"))
        .collect())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::Gemini;
    use std::{
        io::{Read, Write},
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        time::Duration,
    };

    fn spawn_upload_server(count: usize) -> (std::net::SocketAddr, Arc<AtomicUsize>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
        let in_flight = Arc::new(AtomicUsize::new(0));
        let max_seen = Arc::new(AtomicUsize::new(0));
        let server_in_flight = in_flight;
        let server_max_seen = max_seen.clone();

        std::thread::spawn(move || {
            for idx in 0..count {
                let Ok((mut stream, _)) = listener.accept() else {
                    return;
                };
                let active_counter = server_in_flight.clone();
                let peak_counter = server_max_seen.clone();
                std::thread::spawn(move || {
                    let mut request = Vec::new();
                    let mut chunk = [0u8; 4096];
                    let body_end = loop {
                        let Ok(n) = stream.read(&mut chunk) else {
                            return;
                        };
                        if n == 0 {
                            return;
                        }
                        request.extend_from_slice(&chunk[..n]);
                        if let Some(head_end) = request.windows(4).position(|w| w == b"\r\n\r\n") {
                            let head = String::from_utf8_lossy(&request[..head_end]).to_lowercase();
                            let length = head
                                .lines()
                                .find_map(|l| l.strip_prefix("content-length:"))
                                .and_then(|v| v.trim().parse::<usize>().ok())
                                .unwrap_or(0);
                            break head_end + 4 + length;
                        }
                    };
                    while request.len() < body_end {
                        let Ok(n) = stream.read(&mut chunk) else {
                            return;
                        };
                        if n == 0 {
                            break;
                        }
                        request.extend_from_slice(&chunk[..n]);
                    }

                    let current = active_counter.fetch_add(1, Ordering::SeqCst) + 1;
                    let _ = peak_counter.fetch_max(current, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(80));
                    active_counter.fetch_sub(1, Ordering::SeqCst);

                    let body = serde_json::json!({
                        "file": {
                            "name": format!("files/gem-{idx}"),
                            "mimeType": "application/pdf",
                            "uri": format!("https://generativelanguage.googleapis.com/v1beta/files/gem-{idx}"),
                            "state": "ACTIVE"
                        }
                    })
                    .to_string();
                    let response = format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                        body.len(),
                    );
                    let _ = stream.write_all(response.as_bytes());
                    let _ = stream.flush();
                });
            }
        });
        (addr, max_seen)
    }

    #[tokio::test]
    async fn resolve_messages_uploads_attachments_with_bounded_concurrency() {
        let (addr, max_seen) = spawn_upload_server(3);

        let mut dir_bytes = [0u8; 8];
        getrandom::fill(&mut dir_bytes).expect("random dir suffix");
        let temp_dir = std::env::temp_dir().join(format!(
            "aither-gemini-att-test-{:x}",
            u64::from_le_bytes(dir_bytes)
        ));
        async_fs::create_dir_all(&temp_dir).await.expect("temp dir");
        let cache_dir = temp_dir.join("cache");

        let mut attachments = Vec::new();
        for idx in 0..3 {
            let file_path = temp_dir.join(format!("doc-{idx}.pdf"));
            async_fs::write(&file_path, format!("data-{idx}"))
                .await
                .expect("write attachment");
            attachments.push(Attachment::new(
                Url::from_file_path(&file_path).expect("file URL"),
                "application/pdf".parse().expect("mime"),
            ));
        }

        let client = Gemini::new("key").with_base_url(format!("http://{addr}/v1beta"));
        let resolved = resolve_messages_with_cache_dir(
            client.config(),
            vec![Message::user("inspect").with_attachments(attachments)],
            &cache_dir,
        )
        .await
        .expect("uploads succeed");

        let _ = async_fs::remove_dir_all(&temp_dir).await;

        assert_eq!(resolved[0].attachments().len(), 3);
        assert!(
            max_seen.load(Ordering::SeqCst) >= 2,
            "expected concurrent uploads, peak={}",
            max_seen.load(Ordering::SeqCst)
        );
    }
}
