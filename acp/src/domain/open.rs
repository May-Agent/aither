//! Open (extensible) enums and `_`-prefixed extension identifiers.
//!
//! ACP v2 marks every extensible enum as "custom or future": values
//! beginning with `_` are implementation-specific extensions, other
//! unknown values are reserved for future ACP variants. [`Open<T>`] wraps
//! a closed enum and sorts every incoming value into [`Open::Known`],
//! [`Open::Custom`], or [`Open::Unknown`] — preserving the raw payload in
//! the two fallback arms so nothing is lost in transit.
//!
//! Two flavours of wire shape are handled through [`Openable`]:
//!
//! - **Bare strings** (`Openable::TAG == None`): `"end_turn"`,
//!   `"allow_once"`. The raw payload preserved is the string itself.
//! - **Tagged objects** (`Openable::TAG == Some(field)`):
//!   `{"type": "tool_call", ...}` or `{"outcome": "selected", ...}`. The
//!   discriminator is read from `field`; the whole object is preserved.
//!
//! In both cases a *known* tag whose payload fails to parse is a hard
//! deserialization error — never demoted to `Unknown`.

use std::borrow::Cow;
use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};
use serde_json::Value;
use thiserror::Error;

/// Error returned when an [`ExtIdent`] name does not start with `_`.
#[derive(Debug, Clone, Error)]
#[error("extension identifier {0:?} must start with `_`")]
pub struct ExtIdentError(pub String);

/// An ACP extension identifier: a string guaranteed to start with `_`.
///
/// Used both for `_`-prefixed JSON-RPC method names (as [`ExtMethod`]) and
/// for `_`-prefixed enum discriminator values inside [`Open::Custom`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExtIdent(Cow<'static, str>);

impl ExtIdent {
    /// Create an extension identifier from a static name.
    ///
    /// # Panics
    ///
    /// Panics if `name` does not start with `_`. For non-static names use
    /// [`try_new`](Self::try_new).
    #[must_use]
    pub const fn new(name: &'static str) -> Self {
        match name.as_bytes() {
            [b'_', ..] => Self(Cow::Borrowed(name)),
            _ => panic!("extension identifiers must start with `_`"),
        }
    }

    /// Create an extension identifier from any string, checking the `_`
    /// prefix at runtime.
    ///
    /// # Errors
    ///
    /// Returns [`ExtIdentError`] if `name` does not start with `_`.
    pub fn try_new(name: impl Into<String>) -> Result<Self, ExtIdentError> {
        let name = name.into();
        if name.starts_with('_') {
            Ok(Self(Cow::Owned(name)))
        } else {
            Err(ExtIdentError(name))
        }
    }

    /// The identifier as a string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ExtIdent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl AsRef<str> for ExtIdent {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl From<ExtIdent> for String {
    fn from(ident: ExtIdent) -> Self {
        ident.0.into_owned()
    }
}

impl TryFrom<String> for ExtIdent {
    type Error = ExtIdentError;

    fn try_from(name: String) -> Result<Self, Self::Error> {
        Self::try_new(name)
    }
}

impl TryFrom<&str> for ExtIdent {
    type Error = ExtIdentError;

    fn try_from(name: &str) -> Result<Self, Self::Error> {
        Self::try_new(name)
    }
}

impl Serialize for ExtIdent {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for ExtIdent {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = String::deserialize(deserializer)?;
        Self::try_new(value).map_err(D::Error::custom)
    }
}

/// A JSON-RPC method name for an ACP extension.
///
/// Alias for [`ExtIdent`]; the `_` prefix required by the protocol is
/// enforced at construction.
pub type ExtMethod = ExtIdent;

/// Error returned when [`ExtMethod`] is built from a non-`_` name.
pub type ExtMethodError = ExtIdentError;

/// How the discriminator of a wire value is located for [`Open`].
///
/// Implemented by the closed enums wrapped in [`Open<T>`] — see the
/// `open_tags!` macro.
pub trait Openable {
    /// The discriminator field carrying the tag for object unions
    /// (`Some("type")`, `Some("outcome")`, ...), or `None` for enums that
    /// are a bare string on the wire.
    const TAG: Option<&'static str>;

    /// The wire tags this crate models. Any other value is `Custom`
    /// (`_`-prefixed) or `Unknown`.
    const KNOWN: &'static [&'static str];
}

/// A value of an extensible ACP enum: a known variant, a custom
/// `_`-prefixed extension, or an unknown (future) variant.
///
/// `Custom` and `Unknown` both preserve the raw [`Value`] — the complete
/// object for tagged unions, the raw string for bare-string enums — so an
/// unrecognized value round-trips verbatim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Open<T> {
    /// A variant this crate models.
    Known(T),
    /// A `_`-prefixed extension value and its raw payload.
    Custom(ExtIdent, Value),
    /// An unrecognized non-`_` tag and its raw payload.
    Unknown(String, Value),
}

impl<T> Open<T> {
    /// Borrows the known variant, if this is one.
    #[must_use]
    pub const fn as_known(&self) -> Option<&T> {
        match self {
            Self::Known(value) => Some(value),
            Self::Custom(..) | Self::Unknown(..) => None,
        }
    }

    /// Extracts the known variant, if this is one.
    #[must_use]
    pub fn into_known(self) -> Option<T> {
        match self {
            Self::Known(value) => Some(value),
            Self::Custom(..) | Self::Unknown(..) => None,
        }
    }

    /// Whether this value is a modeled variant.
    #[must_use]
    pub const fn is_known(&self) -> bool {
        self.as_known().is_some()
    }

    /// The discriminator string for `Custom`/`Unknown` arms, `None` for
    /// `Known`.
    #[must_use]
    pub fn tag(&self) -> Option<&str> {
        match self {
            Self::Known(_) => None,
            Self::Custom(ident, _) => Some(ident.as_str()),
            Self::Unknown(tag, _) => Some(tag.as_str()),
        }
    }

    /// Maps the known variant, preserving `Custom`/`Unknown`.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Open<U> {
        match self {
            Self::Known(value) => Open::Known(f(value)),
            Self::Custom(ident, raw) => Open::Custom(ident, raw),
            Self::Unknown(tag, raw) => Open::Unknown(tag, raw),
        }
    }

    /// `true` when the value is a `_`-prefixed extension.
    #[must_use]
    pub const fn is_custom(&self) -> bool {
        matches!(self, Self::Custom(..))
    }
}

impl<T: Openable + for<'de> Deserialize<'de>> Open<T> {
    /// Build an `Open<T>` from a raw wire value.
    ///
    /// # Errors
    ///
    /// Returns an error message when the value does not carry the expected
    /// discriminator, or when a *known* tag fails its payload schema.
    /// Unknown tags never error — they become [`Open::Custom`] or
    /// [`Open::Unknown`].
    pub fn from_wire_value(value: Value) -> Result<Self, String> {
        let tag = match T::TAG {
            Some(field) => value
                .get(field)
                .and_then(Value::as_str)
                .map(str::to_owned)
                .ok_or_else(|| format!("missing `{field}` discriminator"))?,
            None => value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| "expected a string enum value".to_string())?,
        };
        Self::from_tag_value(tag, value)
    }

    /// Build an `Open<T>` from an already-extracted tag plus raw value.
    ///
    /// # Errors
    ///
    /// Same contract as [`from_wire_value`](Self::from_wire_value).
    pub fn from_tag_value(tag: String, value: Value) -> Result<Self, String> {
        if T::KNOWN.contains(&tag.as_str()) {
            return serde_json::from_value(value)
                .map(Self::Known)
                .map_err(|error| format!("malformed `{tag}` payload: {error}"));
        }
        if tag.starts_with('_') {
            let ident = ExtIdent::try_new(tag).map_err(|error| error.to_string())?;
            return Ok(Self::Custom(ident, value));
        }
        Ok(Self::Unknown(tag, value))
    }
}

impl<T> From<T> for Open<T> {
    fn from(value: T) -> Self {
        Self::Known(value)
    }
}

impl<T: Serialize> Serialize for Open<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Known(value) => value.serialize(serializer),
            Self::Custom(_, raw) | Self::Unknown(_, raw) => raw.serialize(serializer),
        }
    }
}

impl<'de, T> Deserialize<'de> for Open<T>
where
    T: Openable + for<'de2> Deserialize<'de2>,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        Self::from_wire_value(value).map_err(D::Error::custom)
    }
}

/// Declare the [`Openable`] wiring for a closed enum.
///
/// ```ignore
/// open_tags!(StopReason; None => ["end_turn", "cancelled", "error"]);
/// open_tags!(Plan; Some("type") => ["items"]);
/// ```
macro_rules! open_tags {
    ($ty:ty; $tag:expr => [$($known:literal),* $(,)?]) => {
        impl $crate::domain::open::Openable for $ty {
            const TAG: Option<&'static str> = $tag;
            const KNOWN: &'static [&'static str] = &[$($known),*];
        }
    };
}

pub(crate) use open_tags;

/// Declare a string-valued wire enum whose unknown values round-trip
/// verbatim through an `Other(String)` variant.
///
/// Prefer [`Open`] wrapping when the `_`/future distinction matters; this
/// macro is for leaf enums where a single catch-all is enough.
#[allow(unused_macros)]
macro_rules! string_enum {
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident {
            $(
                $(#[$vmeta:meta])*
                $variant:ident = $value:literal
            ),* $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Eq)]
        $vis enum $name {
            $(
                $(#[$vmeta])*
                $variant,
            )*
            /// A value this crate does not model, preserved verbatim.
            Other(::std::string::String),
        }

        impl $name {
            /// The wire string for this value.
            #[must_use]
            pub const fn as_str(&self) -> &str {
                match self {
                    $(Self::$variant => $value,)*
                    Self::Other(other) => other.as_str(),
                }
            }

            /// Build from a wire string; unknown values become `Other`.
            #[must_use]
            pub fn from_wire(value: ::std::string::String) -> Self {
                $(if value == $value {
                    return Self::$variant;
                })*
                Self::Other(value)
            }
        }

        impl ::std::fmt::Display for $name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                f.write_str(self.as_str())
            }
        }

        impl ::serde::Serialize for $name {
            fn serialize<S: ::serde::Serializer>(
                &self,
                serializer: S,
            ) -> ::std::result::Result<S::Ok, S::Error> {
                serializer.serialize_str(self.as_str())
            }
        }

        impl<'de> ::serde::Deserialize<'de> for $name {
            fn deserialize<D: ::serde::Deserializer<'de>>(
                deserializer: D,
            ) -> ::std::result::Result<Self, D::Error> {
                let value =
                    <::std::string::String as ::serde::Deserialize>::deserialize(deserializer)?;
                ::std::result::Result::Ok(Self::from_wire(value))
            }
        }

        impl ::std::convert::From<::std::string::String> for $name {
            fn from(value: ::std::string::String) -> Self {
                Self::from_wire(value)
            }
        }

        impl ::std::convert::From<&str> for $name {
            fn from(value: &str) -> Self {
                Self::from_wire(value.to_string())
            }
        }
    };
}

#[allow(unused_imports)]
pub(crate) use string_enum;

/// Implements [`Serialize`]/[`Deserialize`] for a union enum whose last two
/// variants are `Custom(ExtIdent, Value)` and `Unknown(String, Value)`.
///
/// Usage inside a module:
///
/// ```ignore
/// open_union_serde!(SessionUpdate, "sessionUpdate", [
///     UserMessageChunk => "user_message_chunk",
///     ToolCallUpdate => "tool_call_update",
/// ]);
/// ```
///
/// Deserialize: a *known* tag is parsed strictly through the payload's own
/// `Deserialize` (a malformed known payload is a hard error). A `_`-tag
/// becomes `Custom`, any other tag becomes `Unknown`, both preserving the
/// raw object. Serialize emits the payload object merged with the
/// discriminator for `Known` variants, and the raw object otherwise.
macro_rules! open_union_serde {
    ($name:ident, $tag:literal, [$($variant:ident => $wire:literal),* $(,)?]) => {
        impl ::serde::Serialize for $name {
            fn serialize<S: ::serde::Serializer>(
                &self,
                serializer: S,
            ) -> ::std::result::Result<S::Ok, S::Error> {
                let mut value = match self {
                    $(
                        Self::$variant(payload) =>
                            ::serde_json::to_value(payload).map_err(::serde::ser::Error::custom)?,
                    )*
                    Self::Custom(_, raw) | Self::Unknown(_, raw) => {
                        return ::serde::Serialize::serialize(raw, serializer);
                    }
                };
                let wire = match self {
                    $(Self::$variant(_) => $wire,)*
                    Self::Custom(..) | Self::Unknown(..) => unreachable!(),
                };
                if let ::serde_json::Value::Object(map) = &mut value {
                    map.insert($tag.to_string(), ::serde_json::Value::from(wire));
                    return ::serde::Serialize::serialize(
                        &::serde_json::Value::Object(map.clone()),
                        serializer,
                    );
                }
                Err(::serde::ser::Error::custom(concat!(
                    stringify!($name),
                    " payload did not serialize to an object"
                )))
            }
        }

        impl<'de> ::serde::Deserialize<'de> for $name {
            fn deserialize<D: ::serde::Deserializer<'de>>(
                deserializer: D,
            ) -> ::std::result::Result<Self, D::Error> {
                let value = <::serde_json::Value as ::serde::Deserialize>::deserialize(deserializer)?;
                let tag = match value.get($tag).and_then(::serde_json::Value::as_str) {
                    Some(tag) => tag.to_owned(),
                    None => {
                        return Err(<D::Error as ::serde::de::Error>::missing_field($tag));
                    }
                };
                match tag.as_str() {
                    $(
                        $wire => ::serde_json::from_value(value)
                            .map(Self::$variant)
                            .map_err(::serde::de::Error::custom),
                    )*
                    _ if tag.starts_with('_') => Ok(Self::Custom(
                        $crate::domain::open::ExtIdent::try_new(tag.clone())
                            .map_err(::serde::de::Error::custom)?,
                        value,
                    )),
                    _ => Ok(Self::Unknown(tag, value)),
                }
            }
        }
    };
}

pub(crate) use open_union_serde;

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    enum Stop {
        EndTurn,
        Cancelled,
        Error,
    }

    open_tags!(Stop; None => ["end_turn", "cancelled", "error"]);

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[serde(tag = "type", rename_all = "snake_case")]
    enum Subject {
        Command { command: String },
    }

    open_tags!(Subject; Some("type") => ["command"]);

    #[test]
    fn open_string_enum_known_custom_unknown() {
        let parsed: Open<Stop> = serde_json::from_value(json!("end_turn")).expect("known");
        assert_eq!(parsed, Open::Known(Stop::EndTurn));

        let parsed: Open<Stop> = serde_json::from_value(json!("_zzz")).expect("custom");
        assert_eq!(parsed, Open::Custom(ExtIdent::new("_zzz"), json!("_zzz")));

        let parsed: Open<Stop> = serde_json::from_value(json!("futuristic")).expect("unknown");
        assert_eq!(
            parsed,
            Open::Unknown("futuristic".to_string(), json!("futuristic"))
        );

        // Round-trips preserve everything.
        for v in [json!("end_turn"), json!("_x"), json!("later")] {
            let parsed: Open<Stop> = serde_json::from_value(v.clone()).expect("parse");
            assert_eq!(serde_json::to_value(&parsed).expect("ser"), v);
        }

        // Non-strings are malformed.
        assert!(serde_json::from_value::<Open<Stop>>(json!(42)).is_err());
    }

    #[test]
    fn open_tagged_union_known_custom_unknown() {
        let parsed: Open<Subject> =
            serde_json::from_value(json!({"type": "command", "command": "ls"})).expect("known");
        assert_eq!(
            parsed,
            Open::Known(Subject::Command {
                command: "ls".to_string()
            })
        );

        let raw = json!({"type": "_mine", "command": "x", "extra": 1});
        let parsed: Open<Subject> = serde_json::from_value(raw.clone()).expect("custom");
        assert!(matches!(parsed, Open::Custom(ref i, _) if i.as_str() == "_mine"));
        assert_eq!(serde_json::to_value(&parsed).expect("ser"), raw);

        let raw = json!({"type": "future_thing", "z": 2});
        let parsed: Open<Subject> = serde_json::from_value(raw.clone()).expect("unknown");
        assert_eq!(parsed, Open::Unknown("future_thing".to_string(), raw));

        // Known tag with malformed payload = hard error.
        assert!(serde_json::from_value::<Open<Subject>>(json!({"type": "command"})).is_err());
        // Missing tag = error.
        assert!(serde_json::from_value::<Open<Subject>>(json!({"x": 1})).is_err());
    }

    #[test]
    fn ext_ident_rules() {
        assert_eq!(ExtIdent::new("_session/goal").as_str(), "_session/goal");
        assert!(ExtIdent::try_new("_x").is_ok());
        assert!(ExtIdent::try_new("session/new").is_err());
        assert_eq!(
            ExtIdent::try_new("nope").expect_err("err").to_string(),
            "extension identifier \"nope\" must start with `_`"
        );
    }

    #[test]
    #[should_panic(expected = "must start with `_`")]
    fn ext_ident_new_panics() {
        let _ = ExtIdent::new("session/new");
    }

    string_enum! {
        /// Test enum.
        pub enum Format {
            /// Git patch.
            GitPatch = "git_patch",
        }
    }

    #[test]
    fn string_enum_captures_other() {
        let parsed: Format = serde_json::from_value(json!("zip")).expect("parse");
        assert_eq!(parsed, Format::Other("zip".to_string()));
        assert_eq!(serde_json::to_value(&parsed).expect("ser"), json!("zip"));
        assert_eq!(Format::from("git_patch"), Format::GitPatch);
    }
}
