//! Exact in-process token counting for models whose vocabulary is published.
//!
//! A [`Vocabulary`] wraps one embedded BPE vocabulary from the `tiktoken`
//! crate — each family lives behind its own feature (`vocab-o200k`,
//! `vocab-cl100k`, `vocab-deepseek`, `vocab-llama`, `vocab-mistral`,
//! `vocab-qwen`), so a client carries only the vocabularies its models can
//! use. [`Vocabulary::for_model`] maps a model id to the published
//! vocabulary it counts with, or `None` for a model with no local counter.

extern crate std;

use alloc::string::ToString;
use std::io::Cursor;

use base64::Engine as _;
use num_traits::ToPrimitive as _;

use crate::llm::LLMRequest;

/// Per-message chat framing, per `OpenAI`'s published recipe: the start,
/// role and end markers of each message.
const TOKENS_PER_MESSAGE: u32 = 3;

/// The primer the model's reply is opened with.
const TOKENS_PER_REPLY: u32 = 3;

/// Image tokens in low-detail mode: a flat cost regardless of size.
const IMAGE_LOW_DETAIL_TOKENS: u32 = 85;

/// Image tokens per 512×512 tile in high-detail mode.
const IMAGE_TILE_TOKENS: u32 = 170;

/// Base cost every high-detail image pays on top of its tiles.
const IMAGE_BASE_TOKENS: u32 = 85;

/// The encoding behind a [`Vocabulary`], for callers that hold the engine
/// themselves.
#[must_use]
pub fn encoding_for_model(model_id: &str) -> Option<&'static str> {
    let id = model_id
        .rsplit('/')
        .next()
        .unwrap_or(model_id)
        .to_ascii_lowercase();
    // OpenAI dates its snapshots "-YYYY-MM-DD"; drop that tail.
    let id = id
        .get(..id.len().saturating_sub(11))
        .filter(|_| {
            id.as_bytes().last().is_some_and(u8::is_ascii_digit)
                && id
                    .as_bytes()
                    .get(id.len().saturating_sub(11))
                    .is_some_and(|b| *b == b'-')
        })
        .unwrap_or(&id);
    if id.starts_with("deepseek") {
        #[cfg(feature = "vocab-deepseek")]
        {
            return Some("deepseek_v3");
        }
    } else if id.starts_with("qwen") {
        #[cfg(feature = "vocab-qwen")]
        {
            return Some("qwen2");
        }
    } else if id.starts_with("llama-3") || id.starts_with("llama3") {
        #[cfg(feature = "vocab-llama")]
        {
            return Some("llama3");
        }
    } else if id.starts_with("mistral")
        || id.starts_with("codestral")
        || id.starts_with("ministral")
        || id.starts_with("devstral")
        || id.starts_with("magistral")
        || id.starts_with("pixtral")
    {
        #[cfg(feature = "vocab-mistral")]
        {
            return Some("mistral_v3");
        }
    } else if id.starts_with("gpt-4o")
        || id.starts_with("gpt-4.1")
        || id.starts_with("gpt-5")
        || id.starts_with('o') && (id.as_bytes().get(1).is_some_and(u8::is_ascii_digit))
        || id.starts_with("gpt-oss")
        || id.starts_with("codex")
    {
        #[cfg(feature = "vocab-o200k")]
        {
            return Some("o200k_base");
        }
    } else if id.starts_with("gpt-4") || id.starts_with("gpt-3.5") {
        #[cfg(feature = "vocab-cl100k")]
        {
            return Some("cl100k_base");
        }
    }
    let _ = id;
    None
}

/// An embedded BPE vocabulary a model counts with, loaded once and held by
/// the client it serves. The engine is a cached `'static` handle, so a
/// `Vocabulary` is a cheap `Copy` handle into it.
#[derive(Clone, Copy)]
pub struct Vocabulary {
    encoding: &'static str,
    bpe: &'static tiktoken::CoreBpe,
}

impl core::fmt::Debug for Vocabulary {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Vocabulary")
            .field("encoding", &self.encoding)
            .finish_non_exhaustive()
    }
}

impl Vocabulary {
    /// The vocabulary `model_id` counts with, or `None` when no published
    /// vocabulary covers it — Claude, Gemini, and ids the table does not
    /// know count through an endpoint or not at all.
    ///
    /// Provider prefixes (`openai/gpt-4o`) and dated snapshot suffixes
    /// (`gpt-4o-2024-08-06`) are stripped before the match; unknown
    /// families fall through to `None` rather than guessing.
    #[must_use]
    pub fn for_model(model_id: &str) -> Option<Self> {
        let encoding = encoding_for_model(model_id)?;
        let bpe = tiktoken::get_encoding(encoding)?;
        Some(Self { encoding, bpe })
    }

    /// The name of the embedded encoding this vocabulary wraps.
    #[must_use]
    pub const fn encoding(&self) -> &'static str {
        self.encoding
    }

    /// The tokens `text` takes in this vocabulary.
    #[must_use]
    pub fn count_text(&self, text: &str) -> u32 {
        u32::try_from(self.bpe.count(text)).unwrap_or(u32::MAX)
    }

    /// The prompt tokens `request` costs: per-message chat framing and the
    /// reply primer from the published recipe, tool-call names and
    /// arguments, tool definitions at their OpenAI-schema size, and images
    /// at the published tile cost.
    ///
    /// Tolerance: the provider's own serialization overhead — role names,
    /// separators and envelope fields the recipe does not publish — so a
    /// count can run a few tokens per message under or over the reported
    /// `prompt_tokens`. Attachments that are not decodable images are not
    /// counted; the budget's usage correction absorbs the rest.
    #[must_use]
    pub fn count_request(&self, request: &LLMRequest) -> u32 {
        let mut total = TOKENS_PER_REPLY;
        for message in request.messages() {
            total += TOKENS_PER_MESSAGE;
            total += self.count_text(message.content());
            for call in message.tool_calls() {
                total += self.count_text(&call.name) + self.count_text(&call.arguments.to_string());
            }
            for attachment in message.attachments() {
                if let Some(tokens) = attachment_tokens(attachment) {
                    total += tokens;
                }
            }
        }
        for definition in request.tool_definitions() {
            total += self.count_text(definition.name())
                + self.count_text(definition.description())
                + self.count_text(&definition.arguments_openai_schema().to_string());
        }
        total
    }
}

/// The tokens one attachment costs, when it is an image whose dimensions
/// can be read: the published high-detail tile formula for the GPT-4o/4.1
/// family. Anything else returns `None` and is not counted.
fn attachment_tokens(attachment: &crate::llm::Attachment) -> Option<u32> {
    let (width, height) = attachment_dimensions(attachment)?;
    Some(image_tokens(width, height))
}

/// Reads image dimensions out of a `data:` URL attachment. Remote or
/// unparseable attachments return `None`.
fn attachment_dimensions(attachment: &crate::llm::Attachment) -> Option<(u32, u32)> {
    if !attachment
        .media_type()
        .type_()
        .as_str()
        .starts_with("image")
    {
        return None;
    }
    let url = attachment.url();
    if url.scheme() != "data" {
        return None;
    }
    let path = url.path();
    let (_, data) = path.split_once(',')?;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data)
        .ok()?;
    image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .ok()?
        .into_dimensions()
        .ok()
}

/// The published GPT-4o/4.1 high-detail formula: scale to fit 2048², then
/// scale the shortest side to 768px, split into 512×512 tiles, and charge
/// base + tile tokens.
fn image_tokens(width: u32, height: u32) -> u32 {
    if width == 0 || height == 0 {
        return IMAGE_LOW_DETAIL_TOKENS;
    }
    let (mut w, mut h) = (f64::from(width), f64::from(height));
    if w > 2048.0 || h > 2048.0 {
        let scale = 2048.0 / w.max(h);
        w *= scale;
        h *= scale;
    }
    let shortest = w.min(h);
    if shortest > 768.0 {
        let scale = 768.0 / shortest;
        w *= scale;
        h *= scale;
    }
    let tiles = (w / 512.0).ceil() * (h / 512.0).ceil();
    IMAGE_BASE_TOKENS + IMAGE_TILE_TOKENS * tiles.to_u32().unwrap_or(u32::MAX)
}

/// `Vocabulary` is shared across threads by the clients that hold it.
const _: () = {
    const fn check<T: Send + Sync>() {}
    check::<Vocabulary>();
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::Message;
    use alloc::format;
    use alloc::vec::Vec;
    use url::Url;

    #[test]
    fn model_ids_map_to_their_published_vocabularies() {
        assert_eq!(encoding_for_model("gpt-4o"), Some("o200k_base"));
        assert_eq!(encoding_for_model("gpt-4o-2024-08-06"), Some("o200k_base"));
        assert_eq!(encoding_for_model("openai/gpt-5.1"), Some("o200k_base"));
        assert_eq!(encoding_for_model("o4-mini"), Some("o200k_base"));
        assert_eq!(encoding_for_model("gpt-4-turbo"), Some("cl100k_base"));
        assert_eq!(encoding_for_model("deepseek-chat"), Some("deepseek_v3"));
        assert_eq!(
            encoding_for_model("deepseek/DeepSeek-R1"),
            Some("deepseek_v3")
        );
        assert_eq!(encoding_for_model("qwen3-coder"), Some("qwen2"));
        assert_eq!(encoding_for_model("llama-3.3-70b"), Some("llama3"));
        assert_eq!(
            encoding_for_model("mistral-large-latest"),
            Some("mistral_v3")
        );
        assert_eq!(encoding_for_model("codestral-latest"), Some("mistral_v3"));
        assert_eq!(encoding_for_model("claude-opus-4-5"), None);
        assert_eq!(encoding_for_model("gemini-3-pro"), None);
        assert_eq!(encoding_for_model("grok-4"), None);
    }

    #[test]
    fn counting_text_uses_the_embedded_vocabulary() {
        let vocab = Vocabulary::for_model("gpt-4o").expect("o200k is published");
        // tiktoken's own reference count for cl100k/o200k family.
        assert_eq!(vocab.count_text("hello world"), 2);
        let cjk = vocab.count_text("中文、日本語、한국어 텍스트를 섞어서 토큰을 세어봅니다");
        assert!(cjk > 0);
        assert_eq!(
            cjk,
            vocab.count_text("中文、日本語、한국어 텍스트를 섞어서 토큰을 세어봅니다")
        );
        assert!(vocab.count_text("hello world") < cjk);
    }

    #[test]
    fn every_vocabulary_family_counts() {
        for model in [
            "gpt-4o",
            "deepseek-r1",
            "qwen2.5",
            "llama-3.1-70b",
            "mistral-large",
        ] {
            let vocab = Vocabulary::for_model(model).expect("published family");
            let text = "Translate this sentence into French";
            let count = vocab.count_text(text);
            assert!(count > 0, "{model}");
            assert_eq!(count, vocab.count_text(text), "{model}");
        }
    }

    #[test]
    fn counting_a_request_charges_framing_tools_and_images() {
        let vocab = Vocabulary::for_model("deepseek-r1").expect("published");
        let request = LLMRequest::new([
            Message::user("describe this screenshot"),
            Message::assistant("it is a chart"),
        ]);
        let bare = vocab.count_request(&LLMRequest::new([Message::user(
            "describe this screenshot",
        )]));
        // Two messages: reply primer + per-message framing + content tokens.
        assert!(bare > 3 + 3, "framing is charged on top of content");
        let two = vocab.count_request(&request);
        assert_eq!(
            two - bare,
            TOKENS_PER_MESSAGE + vocab.count_text("it is a chart")
        );
    }

    /// A complete PNG: signature, IHDR, one IDAT holding a stored deflate
    /// block, and IEND — enough structure for `into_dimensions`.
    fn png_header(width: u32, height: u32) -> Vec<u8> {
        fn crc32(bytes: &[u8]) -> u32 {
            let mut crc = !0_u32;
            for byte in bytes {
                crc ^= u32::from(*byte);
                for _ in 0..8 {
                    crc = if crc & 1 == 1 {
                        crc >> 1 ^ 0xEDB8_8320
                    } else {
                        crc >> 1
                    };
                }
            }
            !crc
        }
        fn adler32(bytes: &[u8]) -> u32 {
            let (mut a, mut b) = (1_u32, 0_u32);
            for byte in bytes {
                a = (a + u32::from(*byte)) % 65521;
                b = (b + a) % 65521;
            }
            b << 16 | a
        }
        fn chunk(kind: [u8; 4], data: &[u8]) -> Vec<u8> {
            let mut chunk = u32::try_from(data.len())
                .expect("test chunks are small")
                .to_be_bytes()
                .to_vec();
            chunk.extend_from_slice(&kind);
            chunk.extend_from_slice(data);
            let mut checked = kind.to_vec();
            checked.extend_from_slice(data);
            chunk.extend_from_slice(&crc32(&checked).to_be_bytes());
            chunk
        }
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&width.to_be_bytes());
        ihdr.extend_from_slice(&height.to_be_bytes());
        ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);
        // zlib: header + stored deflate block wrapping 5 bytes + adler32.
        let scanline = [0, 0, 0, 0, 0];
        let mut idat = Vec::from(&[0x78, 0x01, 0x01][..]);
        let len16 = u16::try_from(scanline.len()).expect("tiny scanline");
        idat.extend_from_slice(&len16.to_le_bytes());
        idat.extend_from_slice(&(!len16).to_le_bytes());
        idat.extend_from_slice(&scanline);
        idat.extend_from_slice(&adler32(&scanline).to_be_bytes());
        let mut bytes = Vec::from(&b"\x89PNG\r\n\x1a\n"[..]);
        bytes.extend_from_slice(&chunk(*b"IHDR", &ihdr));
        bytes.extend_from_slice(&chunk(*b"IDAT", &idat));
        bytes.extend_from_slice(&chunk(*b"IEND", &[]));
        bytes
    }

    #[test]
    fn image_attachments_cost_their_tiles() {
        let data = format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(png_header(1024, 768))
        );
        let attachment = crate::llm::Attachment::new(
            Url::parse(&data).expect("data url parses"),
            "image/png".parse().expect("mime"),
        );
        // 1024×768: shortest side already 768, so 2×2 tiles → 85 + 4×170.
        assert_eq!(attachment_tokens(&attachment), Some(765));
        let remote = crate::llm::Attachment::new(
            Url::parse("https://example.com/shot.png").expect("url"),
            "image/png".parse().expect("mime"),
        );
        assert_eq!(attachment_tokens(&remote), None);
        let pdf = crate::llm::Attachment::new(
            Url::parse(&data).expect("data url parses"),
            "application/pdf".parse().expect("mime"),
        );
        assert_eq!(attachment_tokens(&pdf), None);
    }
}
