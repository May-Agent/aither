use aither_core::image::{Data as ImageData, ImageGenerator, Prompt, Size};
use futures_core::Stream;

use crate::{
    client::call_generate,
    config::Gemini,
    error::GeminiError,
    llm::format_prompt_feedback,
    types::{
        GeminiContent, GenerateContentRequest, GenerateContentResponse, GenerationConfig,
        ImageConfig, Part,
    },
};

impl ImageGenerator for Gemini {
    type Error = GeminiError;

    fn create(
        &self,
        prompt: Prompt,
        size: Size,
    ) -> impl Stream<Item = Result<ImageData, Self::Error>> + Send {
        let cfg = self.config().clone();
        let text = prompt.text().to_owned();
        let images = prompt.images().to_vec();
        let aspect_ratio = map_size_to_aspect_ratio(size);

        async_stream::stream! {
            let Some(model) = cfg.image_model.clone() else {
                yield Err(GeminiError::Api(
                    "image generation is disabled for this Gemini backend".into(),
                ));
                return;
            };
            let mut parts = vec![Part::text(text)];
            for image in images {
                parts.push(Part::inline_image(image));
            }

            let request = GenerateContentRequest {
                system_instruction: None,
                contents: vec![GeminiContent::with_parts("user", parts)],
                generation_config: Some(GenerationConfig {
                    image_config: Some(ImageConfig {
                        aspect_ratio: Some(aspect_ratio),
                        image_size: None,
                    }),
                    ..GenerationConfig::default()
                }),
                tools: Vec::new(),
                tool_config: None,
                safety_settings: Vec::new(),
                cached_content: None,
            };
            match call_generate(&cfg, &model, request)
                .await
                .and_then(|resp| extract_images_from_response(&resp))
            {
                Ok(decoded) => {
                    for image in decoded {
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
    ) -> impl Stream<Item = Result<ImageData, Self::Error>> + Send {
        let cfg = self.config().clone();
        let mask_bytes = mask.to_vec();
        let text = prompt.text().to_owned();
        let base_image = prompt.images().first().cloned();

        async_stream::stream! {
            let Some(model) = cfg.image_model.clone() else {
                yield Err(GeminiError::Api(
                    "image generation is disabled for this Gemini backend".into(),
                ));
                return;
            };
            let Some(base_image) = base_image else {
                yield Err(GeminiError::Api(
                    "image editing requires Prompt::with_image to supply a base image".into(),
                ));
                return;
            };
            let mut parts = vec![Part::inline_image(base_image)];
            if !mask_bytes.is_empty() {
                parts.push(Part::inline_mask(mask_bytes));
            }
            parts.push(Part::text(format!(
                "Apply the mask and follow these instructions: {text}"
            )));
            let request = GenerateContentRequest {
                system_instruction: None,
                contents: vec![GeminiContent::with_parts("user", parts)],
                generation_config: Some(GenerationConfig::default()),
                tools: Vec::new(),
                tool_config: None,
                safety_settings: Vec::new(),
                cached_content: None,
            };
            match call_generate(&cfg, &model, request)
                .await
                .and_then(|resp| extract_images_from_response(&resp))
            {
                Ok(decoded) => {
                    for image in decoded {
                        yield Ok(image);
                    }
                }
                Err(err) => yield Err(err),
            }
        }
    }
}

fn extract_images_from_response(
    response: &GenerateContentResponse,
) -> Result<Vec<ImageData>, GeminiError> {
    let Some(candidate) = response.primary_candidate() else {
        if let Some(feedback) = &response.prompt_feedback {
            return Err(GeminiError::Api(format_prompt_feedback(feedback)));
        }
        return Err(GeminiError::Api(
            "Gemini image response contained no candidates".into(),
        ));
    };
    let Some(content) = &candidate.content else {
        return Err(GeminiError::Api(
            "Gemini image response contained no content parts".into(),
        ));
    };
    let mut images = Vec::new();
    for part in &content.parts {
        if let Some(inline) = &part.inline_data {
            images.push(inline.decode()?);
        }
    }
    if images.is_empty() {
        return Err(GeminiError::Api(
            "Gemini image response contained no inline image data".into(),
        ));
    }
    Ok(images)
}

fn map_size_to_aspect_ratio(size: Size) -> String {
    if size.width() == size.height() {
        "1:1".into()
    } else if size.width() > size.height() {
        if size.width() * 9 == size.height() * 16 {
            "16:9".into()
        } else if size.width() * 3 == size.height() * 4 {
            "4:3".into()
        } else {
            "16:9".into() // Default landscape
        }
    } else if size.width() * 16 == size.height() * 9 {
        "9:16".into()
    } else if size.width() * 4 == size.height() * 3 {
        "3:4".into()
    } else {
        "9:16".into() // Default portrait
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_lite::StreamExt;
    use std::io::{Read, Write};

    fn serve_gemini_image(status: u16, body: &'static str) -> String {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().expect("addr");
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
    async fn create_propagates_disabled_http_and_decode_errors() {
        let disabled = Gemini::new("key").without_image_model();
        let stream = disabled.create(Prompt::new("sunrise"), Size::square(512));
        futures_lite::pin!(stream);
        let err = stream
            .next()
            .await
            .expect("stream yields")
            .expect_err("disabled image model must error");
        assert!(err.to_string().contains("disabled"));

        let err_url = serve_gemini_image(
            403,
            r#"{"error":{"code":403,"message":"forbidden","status":"PERMISSION_DENIED"}}"#,
        );
        let client = Gemini::new("key").with_base_url(err_url);
        let stream = client.create(Prompt::new("sunrise"), Size::square(512));
        futures_lite::pin!(stream);
        let err = stream
            .next()
            .await
            .expect("stream yields")
            .expect_err("HTTP 403 must propagate");
        assert!(matches!(err, GeminiError::Http(_)), "{err:?}");

        let bad_b64_url = serve_gemini_image(
            200,
            r#"{"candidates":[{"content":{"role":"model","parts":[{"inlineData":{"mimeType":"image/png","data":"***invalid***"}}]}}]}"#,
        );
        let client = Gemini::new("key").with_base_url(bad_b64_url);
        let stream = client.create(Prompt::new("sunrise"), Size::square(512));
        futures_lite::pin!(stream);
        let err = stream
            .next()
            .await
            .expect("stream yields")
            .expect_err("invalid base64 must propagate");
        assert!(matches!(err, GeminiError::Decode(_)), "{err:?}");
    }

    #[tokio::test]
    async fn edit_propagates_missing_base_image_and_decodes_valid_image() {
        let client = Gemini::new("key");
        let stream = client.edit(Prompt::new("inpaint"), &[]);
        futures_lite::pin!(stream);
        let err = stream
            .next()
            .await
            .expect("stream yields")
            .expect_err("missing base image must error");
        assert!(err.to_string().contains("base image"));

        let ok_url = serve_gemini_image(
            200,
            r#"{"candidates":[{"content":{"role":"model","parts":[{"inlineData":{"mimeType":"image/png","data":"AQIDBA=="}}]}}]}"#,
        );
        let client = Gemini::new("key").with_base_url(ok_url);
        let stream = client.edit(Prompt::new("inpaint").with_image(vec![9, 9]), &[1]);
        futures_lite::pin!(stream);
        let bytes = stream
            .next()
            .await
            .expect("stream yields")
            .expect("valid image");
        assert_eq!(bytes, vec![1, 2, 3, 4]);
    }
}
