use crate::{
    client::{Config, OpenAI},
    error::OpenAIError,
    files::encode_multipart_parts,
};
use aither_core::image::{Data, ImageGenerator, Prompt, Size};
use base64::{Engine as _, engine::general_purpose};
use futures_core::Stream;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use zenwave::{Client, client, header, multipart::MultipartPart};

impl ImageGenerator for OpenAI {
    type Error = OpenAIError;

    fn create(
        &self,
        prompt: Prompt,
        size: Size,
    ) -> impl Stream<Item = Result<Data, Self::Error>> + Send {
        let cfg = self.config();
        let prompt_text = prompt.text().to_owned();
        let size_token = format_size(size);
        async_stream::stream! {
            match generate_images(cfg, prompt_text, size_token).await {
                Ok(images) => {
                    for image in images {
                        yield Ok(image);
                    }
                }
                Err(err) => yield Err(err),
            }
        }
    }

    fn edit(
        &self,
        prompt: Prompt,
        mask: &[u8],
    ) -> impl Stream<Item = Result<Data, Self::Error>> + Send {
        let cfg = self.config();
        let prompt_text = prompt.text().to_owned();
        let size_token = format_size(Size::square(1024));
        let base = prompt.images().first().cloned();
        let mask_bytes = mask.to_vec();
        async_stream::stream! {
            let Some(base_image) = base else {
                yield Err(OpenAIError::Api(
                    "image editing requires a base image via Prompt::with_image".into(),
                ));
                return;
            };
            match edit_image(cfg, prompt_text, size_token, base_image, mask_bytes).await {
                Ok(images) => {
                    for image in images {
                        yield Ok(image);
                    }
                }
                Err(err) => yield Err(err),
            }
        }
    }
}

async fn generate_images(
    cfg: Arc<Config>,
    prompt: String,
    size: String,
) -> Result<Vec<Data>, OpenAIError> {
    let endpoint = cfg.request_url("/images/generations");
    let mut backend = client();
    let mut builder = backend
        .post(endpoint)
        .map_err(OpenAIError::from_http)?
        .header(header::AUTHORIZATION.as_str(), cfg.request_auth())
        .map_err(OpenAIError::from_http)?
        .header(header::USER_AGENT.as_str(), "aither-openai/0.1")
        .map_err(OpenAIError::from_http)?;
    if let Some(org) = &cfg.organization {
        builder = builder
            .header("OpenAI-Organization", org.clone())
            .map_err(OpenAIError::from_http)?;
    }
    for (name, value) in &cfg.extra_headers {
        builder = builder
            .header(name.clone(), value.clone())
            .map_err(OpenAIError::from_http)?;
    }
    let request = ImageGenerationRequest {
        model: &cfg.image_model,
        prompt: &prompt,
        size: &size,
        response_format: "b64_json",
        n: 1,
    };
    let response: ImageResponse = builder
        .json_body(&request)
        .map_err(OpenAIError::from_http)?
        .json()
        .await
        .map_err(OpenAIError::from_http)?;
    response.into_images()
}

async fn edit_image(
    cfg: Arc<Config>,
    prompt: String,
    size: String,
    image: Vec<u8>,
    mask: Vec<u8>,
) -> Result<Vec<Data>, OpenAIError> {
    let endpoint = cfg.request_url("/images/edits");
    let mut backend = client();
    let mut builder = backend
        .post(endpoint)
        .map_err(OpenAIError::from_http)?
        .header(header::AUTHORIZATION.as_str(), cfg.request_auth())
        .map_err(OpenAIError::from_http)?
        .header(header::USER_AGENT.as_str(), "aither-openai/0.1")
        .map_err(OpenAIError::from_http)?;
    if let Some(org) = &cfg.organization {
        builder = builder
            .header("OpenAI-Organization", org.clone())
            .map_err(OpenAIError::from_http)?;
    }
    for (name, value) in &cfg.extra_headers {
        builder = builder
            .header(name.clone(), value.clone())
            .map_err(OpenAIError::from_http)?;
    }
    let mut parts = vec![
        MultipartPart::text("model", cfg.image_model.clone()),
        MultipartPart::text("prompt", prompt),
        MultipartPart::text("size", size),
        MultipartPart::text("response_format", "b64_json"),
        MultipartPart::binary("image", "image.png", "application/octet-stream", image),
    ];
    if !mask.is_empty() {
        parts.push(MultipartPart::binary(
            "mask",
            "mask.png",
            "application/octet-stream",
            mask,
        ));
    }
    let (boundary, body) = encode_multipart_parts(parts)?;
    let response: ImageResponse = builder
        .header(
            header::CONTENT_TYPE.as_str(),
            format!("multipart/form-data; boundary={boundary}"),
        )
        .map_err(OpenAIError::from_http)?
        .bytes_body(body)
        .json()
        .await
        .map_err(OpenAIError::from_http)?;
    response.into_images()
}

fn format_size(size: Size) -> String {
    format!("{}x{}", size.width(), size.height())
}

#[derive(Debug, Serialize)]
struct ImageGenerationRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    size: &'a str,
    #[serde(rename = "response_format")]
    response_format: &'static str,
    n: u8,
}

#[derive(Debug, Deserialize)]
struct ImageResponse {
    data: Vec<ImagePayload>,
}

impl ImageResponse {
    fn into_images(self) -> Result<Vec<Data>, OpenAIError> {
        if self.data.is_empty() {
            return Err(OpenAIError::Api(
                "image response contained no images".into(),
            ));
        }
        self.data
            .into_iter()
            .map(ImagePayload::into_bytes)
            .collect()
    }
}

#[derive(Debug, Deserialize)]
struct ImagePayload {
    #[serde(default)]
    b64_json: Option<String>,
}

impl ImagePayload {
    fn into_bytes(self) -> Result<Data, OpenAIError> {
        let encoded = self
            .b64_json
            .ok_or_else(|| OpenAIError::Api("image response missing `b64_json` field".into()))?;
        general_purpose::STANDARD
            .decode(encoded)
            .map_err(OpenAIError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_lite::StreamExt;
    use std::io::{Read, Write};

    fn serve_image_response(status: u16, body: &'static str) -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind stub server");
        let addr = listener.local_addr().expect("stub server address");
        std::thread::spawn(move || {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
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
                        .find_map(|line| line.strip_prefix("content-length:"))
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
            let reason = if status == 200 { "OK" } else { "Error" };
            let response = format!(
                "HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len(),
            );
            let _ = stream.write_all(response.as_bytes());
            let _ = stream.flush();
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn create_propagates_http_and_decode_errors() {
        let err_url = serve_image_response(
            429,
            r#"{"error":{"message":"slow down","type":"requests","code":"rate_limit_exceeded"}}"#,
        );
        let client = OpenAI::builder("key").base_url(err_url).build();
        let stream = client.create(Prompt::new("a lighthouse"), Size::square(512));
        futures_lite::pin!(stream);
        let err = stream
            .next()
            .await
            .expect("stream yields")
            .expect_err("HTTP 429 must propagate");
        assert!(matches!(err, OpenAIError::RateLimit { .. }), "{err:?}");

        let bad_b64_url =
            serve_image_response(200, r#"{"data":[{"b64_json":"!!!not-base64!!!"}]}"#);
        let client = OpenAI::builder("key").base_url(bad_b64_url).build();
        let stream = client.create(Prompt::new("a lighthouse"), Size::square(512));
        futures_lite::pin!(stream);
        let err = stream
            .next()
            .await
            .expect("stream yields")
            .expect_err("invalid base64 must propagate");
        assert!(matches!(err, OpenAIError::Decode(_)), "{err:?}");

        let empty_url = serve_image_response(200, r#"{"data":[]}"#);
        let client = OpenAI::builder("key").base_url(empty_url).build();
        let stream = client.create(Prompt::new("a lighthouse"), Size::square(512));
        futures_lite::pin!(stream);
        let err = stream
            .next()
            .await
            .expect("stream yields")
            .expect_err("empty image list must propagate");
        assert!(matches!(err, OpenAIError::Api(_)), "{err:?}");
    }

    #[tokio::test]
    async fn edit_propagates_missing_base_image_and_upstream_errors() {
        let client = OpenAI::new("key");
        let stream = client.edit(Prompt::new("remove clouds"), &[]);
        futures_lite::pin!(stream);
        let err = stream
            .next()
            .await
            .expect("stream yields")
            .expect_err("missing base image must fail");
        assert!(err.to_string().contains("base image"));

        let ok_url = serve_image_response(200, r#"{"data":[{"b64_json":"AQID"}]}"#);
        let client = OpenAI::builder("key").base_url(ok_url).build();
        let stream = client.edit(Prompt::new("add sun").with_image(vec![9, 8, 7]), &[1, 2]);
        futures_lite::pin!(stream);
        let bytes = stream
            .next()
            .await
            .expect("stream yields")
            .expect("valid image payload");
        assert_eq!(bytes, vec![1, 2, 3]);
    }
}
