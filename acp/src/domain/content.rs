//! Content blocks and message payloads.
//!
//! [`ContentBlock`] is an open union tagged `type`: `text`, `image`,
//! `audio`, `resource_link`, `resource`, plus `Custom`/`Unknown` arms that
//! preserve unrecognized blocks verbatim.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::ids::MessageId;
use super::open::{ExtIdent, Open, open_union_serde};

/// A unit of message content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentBlock {
    /// Text content (`"type": "text"`).
    Text(TextContent),
    /// Image content (`"type": "image"`).
    Image(ImageContent),
    /// Audio content (`"type": "audio"`).
    Audio(AudioContent),
    /// A link to a resource (`"type": "resource_link"`).
    ResourceLink(ResourceLink),
    /// An embedded resource (`"type": "resource"`).
    Resource(EmbeddedResource),
    /// A `_`-prefixed extension block, preserved verbatim.
    Custom(ExtIdent, Value),
    /// An unrecognized block type, preserved verbatim.
    Unknown(String, Value),
}

open_union_serde!(ContentBlock, "type", [
    Text => "text",
    Image => "image",
    Audio => "audio",
    ResourceLink => "resource_link",
    Resource => "resource",
]);

impl ContentBlock {
    /// A plain text block.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(TextContent {
            text: text.into(),
            annotations: None,
            meta: None,
            extra: BTreeMap::new(),
        })
    }

    /// Extracts the plain text of a [`ContentBlock::Text`], else `None`.
    #[must_use]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(&text.text),
            _ => None,
        }
    }
}

impl From<&str> for ContentBlock {
    fn from(text: &str) -> Self {
        Self::text(text)
    }
}

impl From<String> for ContentBlock {
    fn from(text: String) -> Self {
        Self::text(text)
    }
}

impl From<TextContent> for ContentBlock {
    fn from(content: TextContent) -> Self {
        Self::Text(content)
    }
}
impl From<ImageContent> for ContentBlock {
    fn from(content: ImageContent) -> Self {
        Self::Image(content)
    }
}
impl From<AudioContent> for ContentBlock {
    fn from(content: AudioContent) -> Self {
        Self::Audio(content)
    }
}
impl From<ResourceLink> for ContentBlock {
    fn from(link: ResourceLink) -> Self {
        Self::ResourceLink(link)
    }
}
impl From<EmbeddedResource> for ContentBlock {
    fn from(resource: EmbeddedResource) -> Self {
        Self::Resource(resource)
    }
}

/// Text content (`{"type": "text", "text": ...}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextContent {
    /// The text.
    pub text: String,
    /// Optional annotations (`Annotations` object; kept as raw JSON).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Value>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Image content (`{"type": "image", "data"|"uri", "mimeType"}`).
///
/// `data` is required by the v2 schema but optional here so v1 blocks that
/// carry only `uri` still upgrade.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageContent {
    /// Base64-encoded image bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    /// Image MIME type.
    pub mime_type: String,
    /// URI carrying the image instead of inline `data`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
    /// Optional annotations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Value>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Audio content (`{"type": "audio", "data", "mimeType"}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioContent {
    /// Base64-encoded audio bytes.
    pub data: String,
    /// Audio MIME type.
    pub mime_type: String,
    /// Optional annotations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Value>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// A link to a resource without embedding its contents
/// (`{"type": "resource_link", "uri", "name", ...}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceLink {
    /// Resource name.
    pub name: String,
    /// Resource URI.
    pub uri: String,
    /// Human-readable title.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Human-readable description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Icons for this resource.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub icons: Vec<Icon>,
    /// MIME type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Size in bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
    /// Optional annotations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Value>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl ResourceLink {
    /// A link with a name and URI.
    #[must_use]
    pub fn new(uri: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            uri: uri.into(),
            title: None,
            description: None,
            icons: Vec::new(),
            mime_type: None,
            size: None,
            annotations: None,
            meta: None,
            extra: BTreeMap::new(),
        }
    }
}

/// An icon (`{"src", "mimeType"?, "sizes"?, "theme"?}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Icon {
    /// Icon source URI.
    pub src: String,
    /// Icon MIME type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Offered sizes (e.g. `"32x32"`, `"any"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sizes: Option<Vec<String>>,
    /// Intended theme (`"light"`/`"dark"`/custom string).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub theme: Option<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// An embedded resource (`{"type": "resource", "resource": {...}}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddedResource {
    /// The resource contents.
    pub resource: ResourceContents,
    /// Optional annotations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Value>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

/// Contents of an [`EmbeddedResource`]: text or base64 blob.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ResourceContents {
    /// Text contents (`{"uri", "text", "mimeType"?}`).
    Text(TextResourceContents),
    /// Base64 blob contents (`{"uri", "blob", "mimeType"?}`).
    Blob(BlobResourceContents),
}

/// Text resource contents.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextResourceContents {
    /// Resource URI.
    pub uri: String,
    /// Text contents.
    pub text: String,
    /// MIME type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Blob resource contents.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BlobResourceContents {
    /// Resource URI.
    pub uri: String,
    /// Base64-encoded binary contents.
    pub blob: String,
    /// MIME type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// A streamed item of message content
/// (`{"messageId", "content"}`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContentChunk {
    /// The message this chunk appends to.
    pub message_id: MessageId,
    /// The content item.
    pub content: ContentBlock,
    /// Chunk-scoped extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl ContentChunk {
    /// A chunk of `content` for `message_id`.
    #[must_use]
    pub fn new(message_id: impl Into<MessageId>, content: impl Into<ContentBlock>) -> Self {
        Self {
            message_id: message_id.into(),
            content: content.into(),
            meta: None,
            extra: BTreeMap::new(),
        }
    }
}

/// A whole-message upsert shared by `user_message`, `agent_message`, and
/// `agent_thought` updates.
///
/// `content` has patch semantics: omitted leaves stored content, `null`
/// clears it, a concrete array replaces it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Message {
    /// The message being created or patched.
    pub message_id: MessageId,
    /// Message content; patch semantics.
    #[serde(default, skip_serializing_if = "super::patch::Patch::is_unchanged")]
    pub content: super::patch::Patch<Vec<ContentBlock>>,
    /// Message-scoped extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

impl Message {
    /// A message carrying `content`.
    #[must_use]
    pub fn new(message_id: impl Into<MessageId>, content: Vec<ContentBlock>) -> Self {
        Self {
            message_id: message_id.into(),
            content: super::patch::Patch::Set(content),
            meta: None,
            extra: BTreeMap::new(),
        }
    }
}

/// Sender/recipient role on an `Annotations` audience.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// Content from the assistant.
    Assistant,
    /// Content from the user.
    User,
}

impl super::open::Openable for Role {
    const TAG: Option<&'static str> = None;
    const KNOWN: &'static [&'static str] = &["assistant", "user"];
}

/// Annotations carried by content blocks (`Annotations` object).
///
/// Kept as a typed convenience; `extra` preserves unknown keys.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Annotations {
    /// Intended audience roles.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audience: Option<Vec<Open<Role>>>,
    /// RFC 3339 last-modified time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_modified: Option<String>,
    /// Priority in `[0, 1]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority: Option<f64>,
    /// Extension metadata.
    #[serde(rename = "_meta", default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
    /// Vendor fields preserved verbatim.
    #[serde(flatten)]
    pub extra: BTreeMap<String, Value>,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn content_block_known_and_fallback_round_trip() {
        let block = ContentBlock::text("hi");
        let json = serde_json::to_value(&block).expect("ser");
        assert_eq!(json["type"], "text");
        assert_eq!(json["text"], "hi");
        let parsed: ContentBlock = serde_json::from_value(json).expect("de");
        assert_eq!(parsed.as_text(), Some("hi"));

        // Custom `_` type preserved verbatim.
        let raw = json!({"type": "_diff_v3", "x": 1});
        let parsed: ContentBlock = serde_json::from_value(raw.clone()).expect("de");
        assert!(matches!(parsed, ContentBlock::Custom(ref i, _) if i.as_str() == "_diff_v3"));
        assert_eq!(serde_json::to_value(&parsed).expect("ser"), raw);

        // Unknown future type preserved verbatim.
        let raw = json!({"type": "hologram", "v": 2});
        let parsed: ContentBlock = serde_json::from_value(raw.clone()).expect("de");
        assert_eq!(parsed, ContentBlock::Unknown("hologram".to_string(), raw));

        // Malformed known payload is a hard error.
        assert!(serde_json::from_value::<ContentBlock>(json!({"type": "text"})).is_err());
        // Missing tag is a hard error.
        assert!(serde_json::from_value::<ContentBlock>(json!({"text": "x"})).is_err());
    }

    #[test]
    fn message_content_patch_semantics() {
        let raw = json!({"messageId": "m1"});
        let parsed: Message = serde_json::from_value(raw).expect("de");
        assert!(parsed.content.is_unchanged());
        let raw = json!({"messageId": "m1", "content": null});
        let parsed: Message = serde_json::from_value(raw).expect("de");
        assert_eq!(parsed.content, super::super::patch::Patch::Clear);
        let raw = json!({"messageId": "m1", "content": [{"type": "text", "text": "a"}]});
        let parsed: Message = serde_json::from_value(raw).expect("de");
        assert_eq!(parsed.content.as_option().map(Vec::len), Some(1));
        assert_eq!(
            serde_json::to_value(&parsed).expect("ser")["content"][0]["type"],
            "text"
        );
    }
}
