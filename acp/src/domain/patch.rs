//! Patch semantics for ACP v2 upsert payloads.
//!
//! Most v2 update objects use *patch* semantics on their optional fields:
//! an omitted key leaves the stored value unchanged, an explicit `null`
//! clears it, and a concrete value replaces it. [`Patch`] models that
//! three-state field, and [`MetaPatch`] models the `_meta` key-patch shape.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

/// A field with ACP v2 patch semantics.
///
/// | Wire value | Variant |
/// |------------|---------|
/// | key absent | [`Patch::Unchanged`] |
/// | `null`     | [`Patch::Clear`] |
/// | `value`    | [`Patch::Set`] `(value)` |
///
/// Serialization skips `Unchanged` fields (via
/// `#[serde(default, skip_serializing_if = "Patch::is_unchanged")]`, which
/// the derived `Deserialize` requires to default missing keys), emits `null`
/// for `Clear`, and emits the wrapped value for `Set`.
///
/// When a `Patch` is serialized on its own — where skipping is impossible —
/// `Unchanged` and `Clear` both emit `null`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum Patch<T> {
    /// The field was not mentioned; keep the stored value.
    #[default]
    Unchanged,
    /// The field was explicitly cleared with `null`.
    Clear,
    /// The field was replaced with a new value.
    Set(T),
}

impl<T> Patch<T> {
    /// Whether this field was omitted from the update.
    #[must_use]
    pub const fn is_unchanged(&self) -> bool {
        matches!(self, Self::Unchanged)
    }

    /// Whether this field carries a value or an explicit clear.
    #[must_use]
    pub const fn is_changed(&self) -> bool {
        !self.is_unchanged()
    }

    /// The set value, or `default` for `Unchanged`/`Clear`.
    pub fn or(self, default: T) -> T {
        match self {
            Self::Set(value) => value,
            Self::Unchanged | Self::Clear => default,
        }
    }

    /// The set value, or the stored `stored` value for `Unchanged`/`Clear`.
    ///
    /// Unlike [`or`](Self::or), `Clear` does *not* produce `stored`: it
    /// yields `None`, modelling an explicit clear.
    pub fn apply_to(self, stored: Option<T>) -> Option<T> {
        match self {
            Self::Set(value) => Some(value),
            Self::Clear => None,
            Self::Unchanged => stored,
        }
    }

    /// Map the contained value, preserving `Unchanged`/`Clear`.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Patch<U> {
        match self {
            Self::Unchanged => Patch::Unchanged,
            Self::Clear => Patch::Clear,
            Self::Set(value) => Patch::Set(f(value)),
        }
    }

    /// `Some(&T)` when `Set`, `None` otherwise.
    #[must_use]
    pub const fn as_option(&self) -> Option<&T> {
        match self {
            Self::Set(value) => Some(value),
            Self::Unchanged | Self::Clear => None,
        }
    }

    /// `Some(T)` when `Set`, `None` otherwise.
    pub fn into_option(self) -> Option<T> {
        match self {
            Self::Set(value) => Some(value),
            Self::Unchanged | Self::Clear => None,
        }
    }

    /// Borrows this patch as a patch of `&T`.
    #[must_use]
    pub const fn as_ref(&self) -> Patch<&T> {
        match self {
            Self::Unchanged => Patch::Unchanged,
            Self::Clear => Patch::Clear,
            Self::Set(value) => Patch::Set(value),
        }
    }
}

impl<T> From<T> for Patch<T> {
    fn from(value: T) -> Self {
        Self::Set(value)
    }
}

impl<T> From<Option<T>> for Patch<T> {
    /// `Some` becomes [`Patch::Set`], `None` becomes [`Patch::Clear`].
    fn from(value: Option<T>) -> Self {
        value.map_or_else(|| Self::Clear, Self::Set)
    }
}

impl<T: Serialize> Serialize for Patch<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            // A lone `Patch` cannot skip itself; `null` is the least
            // surprising encoding for both empty states. Struct fields use
            // `skip_serializing_if` so `Unchanged` never reaches this path.
            Self::Unchanged | Self::Clear => serializer.serialize_none(),
            Self::Set(value) => value.serialize(serializer),
        }
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Patch<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Reached only when the key is present (fields pair this with
        // `#[serde(default)]`): `null` -> Clear, value -> Set.
        Option::<T>::deserialize(deserializer)?
            .map_or_else(|| Ok(Self::Clear), |value| Ok(Self::Set(value)))
    }
}

/// A patch applied to a `_meta` object, key by key.
///
/// Serializes as a JSON object; keys mapped to `None` emit `null` (the
/// v2 per-key clear), keys mapped to `Some(value)` emit the value.
/// An empty map serializes as `{}`.
///
/// This is the per-key variant of patch semantics for `_meta` fields on
/// update objects that merge rather than replace their metadata.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MetaPatch(pub BTreeMap<String, Option<Value>>);

impl MetaPatch {
    /// An empty patch.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether the patch carries no keys (serializes `{}`).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Set key `key` to `value`.
    #[must_use]
    pub fn set(mut self, key: impl Into<String>, value: Value) -> Self {
        self.0.insert(key.into(), Some(value));
        self
    }

    /// Clear key `key` (emits `null`).
    #[must_use]
    pub fn clear(mut self, key: impl Into<String>) -> Self {
        self.0.insert(key.into(), None);
        self
    }
}

impl From<BTreeMap<String, Option<Value>>> for MetaPatch {
    fn from(map: BTreeMap<String, Option<Value>>) -> Self {
        Self(map)
    }
}

impl Serialize for MetaPatch {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for MetaPatch {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        BTreeMap::<String, Option<Value>>::deserialize(deserializer).map(Self)
    }
}

/// An empty-object capability/support marker.
///
/// ACP v2 advertises supported features as `{}` presence markers: omitted
/// (or `null`) means unsupported, an object — even an empty one — means
/// supported. `EmptyObject` is the unit payload for those markers: it always
/// serializes as `{}` and accepts any JSON object on deserialize.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct EmptyObject;

impl Serialize for EmptyObject {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap as _;
        serializer.serialize_map(Some(0))?.end()
    }
}

impl<'de> Deserialize<'de> for EmptyObject {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;

        impl<'de2> serde::de::Visitor<'de2> for Visitor {
            type Value = EmptyObject;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("an object")
            }

            fn visit_map<A: serde::de::MapAccess<'de2>>(
                self,
                mut map: A,
            ) -> Result<EmptyObject, A::Error> {
                while map
                    .next_entry::<serde::de::IgnoredAny, serde::de::IgnoredAny>()?
                    .is_some()
                {}
                Ok(EmptyObject)
            }
        }

        deserializer.deserialize_map(Visitor)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[derive(Debug, Default, Serialize, Deserialize, PartialEq)]
    #[serde(rename_all = "camelCase")]
    struct Holder {
        #[serde(default, skip_serializing_if = "Patch::is_unchanged")]
        field: Patch<String>,
        #[serde(default, skip_serializing_if = "Patch::is_unchanged")]
        meta: Patch<Value>,
    }

    #[test]
    fn patch_three_states_round_trip() {
        // Absent -> Unchanged, and Unchanged skips itself.
        let parsed: Holder = serde_json::from_value(json!({})).expect("deserialize");
        assert_eq!(parsed, Holder::default());
        assert_eq!(serde_json::to_value(&parsed).expect("serialize"), json!({}));

        // null -> Clear, and Clear serializes as null.
        let parsed: Holder = serde_json::from_value(json!({"field": null})).expect("deserialize");
        assert_eq!(parsed.field, Patch::Clear);
        assert_eq!(parsed.meta, Patch::Unchanged);
        let back = serde_json::to_value(&parsed).expect("serialize");
        assert_eq!(back, json!({"field": null}));

        // value -> Set.
        let parsed: Holder =
            serde_json::from_value(json!({"field": "x", "meta": {"a": 1}})).expect("deserialize");
        assert_eq!(parsed.field, Patch::Set("x".to_string()));
        assert_eq!(parsed.meta, Patch::Set(json!({"a": 1})));
        let back = serde_json::to_value(&parsed).expect("serialize");
        assert_eq!(back, json!({"field": "x", "meta": {"a": 1}}));
    }

    #[test]
    fn patch_helpers() {
        assert_eq!(Patch::Set(1).or(2), 1);
        assert_eq!(Patch::<i32>::Clear.or(2), 2);
        assert_eq!(Patch::<i32>::Unchanged.or(2), 2);

        assert_eq!(Patch::Set(1).apply_to(Some(9)), Some(1));
        assert_eq!(Patch::<i32>::Clear.apply_to(Some(9)), None);
        assert_eq!(Patch::<i32>::Unchanged.apply_to(Some(9)), Some(9));

        assert_eq!(Patch::Set(2).map(|v| v * 3), Patch::Set(6));
        assert_eq!(Patch::<i32>::Clear.map(|v| v * 3), Patch::Clear);

        assert_eq!(
            Patch::Set("a".to_string()).as_option().map(String::as_str),
            Some("a")
        );
        assert!(Patch::is_unchanged(&Patch::<i32>::Unchanged));
    }

    #[test]
    fn meta_patch_serializes_null_per_key() {
        let patch = MetaPatch::new().set("keep", json!(1)).clear("gone");
        let back = serde_json::to_value(&patch).expect("serialize");
        assert_eq!(back, json!({"keep": 1, "gone": null}));

        let parsed: MetaPatch = serde_json::from_value(back).expect("deserialize");
        assert_eq!(parsed.0["keep"], Some(json!(1)));
        assert_eq!(parsed.0["gone"], None);
    }

    #[test]
    fn empty_object_serializes_as_empty_map() {
        assert_eq!(
            serde_json::to_value(EmptyObject).expect("serialize"),
            json!({})
        );
        let parsed: EmptyObject =
            serde_json::from_value(json!({"any": "thing"})).expect("deserialize");
        assert_eq!(parsed, EmptyObject);
    }
}
