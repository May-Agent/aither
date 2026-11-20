//! Typed identifiers used across the protocol.
//!
//! Transparent `String` newtypes; `new()` allocates a UUID v4 where the
//! peer that mints the id normally generates it (messages, tool calls,
//! terminals, plans, compactions, elicitations, sessions).

use std::fmt;

use serde::{Deserialize, Serialize};

macro_rules! id_type {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident $(; $gen:ident)?
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        #[serde(transparent)]
        #[repr(transparent)]
        $vis struct $name(String);

        impl $name {
            /// Wraps an existing id string.
            #[must_use]
            pub fn from_string(id: impl Into<String>) -> Self {
                Self(id.into())
            }

            /// The id as a string slice.
            #[must_use]
            pub fn as_str(&self) -> &str {
                &self.0
            }

            /// The id as an owned `String`.
            #[must_use]
            pub fn into_string(self) -> String {
                self.0
            }

            id_type!(@gen $name $($gen)?);
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                &self.0
            }
        }

        impl From<String> for $name {
            fn from(id: String) -> Self {
                Self(id)
            }
        }

        impl From<&str> for $name {
            fn from(id: &str) -> Self {
                Self(id.to_string())
            }
        }

        impl From<$name> for String {
            fn from(id: $name) -> Self {
                id.0
            }
        }
    };
    (@gen $name:ident uuid) => {
        /// Allocates a fresh random (UUID v4) id.
        #[must_use]
        pub fn new() -> Self {
            Self(uuid::Uuid::new_v4().to_string())
        }
    };
    (@gen $name:ident) => {};
}

id_type! {
    /// Identifies a session between a client and an agent.
    pub struct SessionId; uuid
}
impl Default for SessionId {
    fn default() -> Self {
        Self::new()
    }
}

id_type! {
    /// Identifies a message, unique among messages of the same type within
    /// a session.
    pub struct MessageId; uuid
}
impl Default for MessageId {
    fn default() -> Self {
        Self::new()
    }
}

id_type! {
    /// Identifies a tool call within a session.
    pub struct ToolCallId; uuid
}
impl Default for ToolCallId {
    fn default() -> Self {
        Self::new()
    }
}

id_type! {
    /// Identifies an agent-owned terminal within a session.
    pub struct TerminalId; uuid
}
impl Default for TerminalId {
    fn default() -> Self {
        Self::new()
    }
}

id_type! {
    /// Identifies a plan within a session.
    pub struct PlanId; uuid
}
impl Default for PlanId {
    fn default() -> Self {
        Self::new()
    }
}

id_type! {
    /// Identifies a context compaction within a session.
    pub struct CompactionId; uuid
}
impl Default for CompactionId {
    fn default() -> Self {
        Self::new()
    }
}

id_type! {
    /// Identifies an in-flight elicitation.
    pub struct ElicitationId; uuid
}
impl Default for ElicitationId {
    fn default() -> Self {
        Self::new()
    }
}

id_type! {
    /// Identifies a session configuration option.
    pub struct ConfigId
}

id_type! {
    /// Identifies a selectable configuration value.
    pub struct ConfigValueId
}

id_type! {
    /// Identifies a configuration-option value group.
    pub struct ConfigGroupId
}

id_type! {
    /// Identifies an authentication method.
    pub struct AuthMethodId
}

id_type! {
    /// Identifies a permission option.
    pub struct PermissionOptionId
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_round_trip_transparent() {
        let id = MessageId::new();
        assert_ne!(id.as_str(), "");
        let json = serde_json::to_value(&id).expect("ser");
        assert_eq!(json, serde_json::Value::from(id.as_str()));
        let back: MessageId = serde_json::from_value(json).expect("de");
        assert_eq!(back, id);
        let s: SessionId = "s-1".into();
        assert_eq!(s.as_str(), "s-1");
        assert_eq!(SessionId::from(String::from(s.clone())), s);
    }
}
