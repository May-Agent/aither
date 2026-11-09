//! Communication and working languages.
//!
//! A host sets the language the agent talks to the user in and the language it
//! writes in for others to read. This module holds the settings
//! ([`Languages`]), the detection of a text's language on its prose and the
//! enforcement ([`LanguageGuard`]) the agent loop and the harness commands
//! share.

mod detect;
mod guard;
mod prose;
mod watch;

use std::fmt;
use std::str::FromStr;

pub use detect::DetectorError;
pub use guard::{LanguageAlert, LanguageGuard, LanguageRefusal, WorkingText};
pub(crate) use guard::{default_alerts, default_refusal, translate};
pub(crate) use watch::LanguageAlerts;

/// A natural language the detector can identify.
///
/// A language is the ISO 639-3 language the detector's labels name, with the
/// variants it tells apart folded into their macrolanguage: Chinese in either
/// script and Cantonese are Chinese, Gulf and Levantine Arabic are Arabic,
/// Dari is Persian.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Language(isolang::Language);

/// A language name the detector does not know.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error(
    "unknown language `{0}`; use an English name, a native name or a language code the detector identifies"
)]
pub struct UnknownLanguage(String);

/// The ISO 639-3 code a detector label's language is known by: the
/// macrolanguage where the label names one of its members.
fn macrolanguage(code: &str) -> &str {
    match code {
        "arb" | "acm" | "acq" | "aeb" | "ajp" | "apc" | "ars" | "ary" | "arz" => "ara",
        "pes" | "prs" => "fas",
        "cmn" | "yue" => "zho",
        "lvs" => "lav",
        "zsm" => "msa",
        "khk" => "mon",
        "npi" => "nep",
        "gaz" => "orm",
        "swh" => "swa",
        "uzn" => "uzb",
        "azj" | "azb" => "aze",
        "als" => "sqi",
        "plt" => "mlg",
        "pbt" => "pus",
        "ydd" => "yid",
        "ayr" => "aym",
        "quy" => "que",
        other => other,
    }
}

impl Language {
    /// English, the default working language.
    pub const ENGLISH: Self = Self(isolang::Language::Eng);
    /// Chinese in either script, Cantonese included.
    pub(crate) const CHINESE: Self = Self(isolang::Language::Zho);
    pub(crate) const JAPANESE: Self = Self(isolang::Language::Jpn);
    pub(crate) const KOREAN: Self = Self(isolang::Language::Kor);

    /// Whether the language is written in Han, kana or Hangul.
    pub(crate) fn is_cjk(self) -> bool {
        self == Self::CHINESE || self == Self::JAPANESE || self == Self::KOREAN
    }

    /// Whether the language is one the Han script alone can leave undecided.
    pub(crate) fn is_chinese_or_japanese(self) -> bool {
        self == Self::CHINESE || self == Self::JAPANESE
    }

    /// The language of a detector label such as `zho_Hans`, if this crate
    /// knows it.
    pub(crate) fn from_label(label: &str) -> Option<Self> {
        let code = label.split_once('_').map_or(label, |(code, _)| code);
        isolang::Language::from_639_3(macrolanguage(code)).map(Self)
    }

    /// Whether the detector can answer with this language.
    fn is_detectable(self) -> bool {
        detect::LABELS
            .lines()
            .any(|label| Self::from_label(label) == Some(self))
    }

    /// The language's English name, such as `Chinese` or `Spanish`.
    #[must_use]
    pub fn name(self) -> &'static str {
        self.0.to_name()
    }

    /// The language's ISO 639-3 code, such as `zho` or `spa`.
    #[must_use]
    pub fn code(self) -> &'static str {
        self.0.to_639_3()
    }

    pub(crate) fn is_english(self) -> bool {
        self == Self::ENGLISH
    }
}

impl fmt::Display for Language {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Language {
    type Err = UnknownLanguage;

    /// Parses an English name (`Chinese`, `german`), a native name or an ISO
    /// 639-1 or 639-3 code (`zh`, `cmn`, `deu`), ignoring case. Only a
    /// language the detector can identify parses.
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let trimmed = text.trim();
        let wanted = trimmed.to_lowercase();
        isolang::Language::from_639_1(&wanted)
            .or_else(|| isolang::Language::from_639_3(macrolanguage(&wanted)))
            .or_else(|| isolang::Language::from_name_lowercase(&wanted))
            .or_else(|| isolang::Language::from_autonym(trimmed))
            .map(Self)
            .filter(|language| language.is_detectable())
            .ok_or_else(|| UnknownLanguage(text.to_string()))
    }
}

/// The languages a session works in.
///
/// The communication language is the one the agent talks to the user in:
/// replies and the questions it asks. The working language, English by
/// default, is the one it writes in for others to read: its working document, prompts
/// and messages to subagents and other agents, issue and pull request text.
/// Both are inactive while the communication language is unset or English,
/// and the working language is enforced only once the communication language
/// is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Languages {
    communication: Option<Language>,
    working: Language,
}

impl Languages {
    /// Talks to the user in `communication` and writes in English.
    #[must_use]
    pub const fn new(communication: Language) -> Self {
        Self {
            communication: Some(communication),
            working: Language::ENGLISH,
        }
    }

    /// No language is set.
    #[must_use]
    pub const fn unset() -> Self {
        Self {
            communication: None,
            working: Language::ENGLISH,
        }
    }

    /// Writes for others in `working` instead of English.
    #[must_use]
    pub const fn with_working(mut self, working: Language) -> Self {
        self.working = working;
        self
    }

    /// The language the user is talked to in, as set. It is also what a
    /// sidecar that writes to the user, such as a recap or a prompt
    /// suggestion, writes in; [`enforced`](Self::enforced) tells whether the
    /// agent itself is held to it.
    #[must_use]
    pub const fn communication(&self) -> Option<Language> {
        self.communication
    }

    /// The language written for others, as set.
    #[must_use]
    pub const fn working(&self) -> Language {
        self.working
    }

    /// Whether languages are enforced: the communication language is set
    /// and is not English.
    #[must_use]
    pub fn enforced(&self) -> bool {
        self.communication
            .is_some_and(|language| !language.is_english())
    }
}

impl Default for Languages {
    fn default() -> Self {
        Self::unset()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn languages_parse_by_name_and_code() {
        assert_eq!("Chinese".parse::<Language>().unwrap().code(), "zho");
        assert_eq!("cmn".parse::<Language>().unwrap().name(), "Chinese");
        assert_eq!("zh".parse::<Language>().unwrap().name(), "Chinese");
        assert_eq!("deu".parse::<Language>().unwrap().name(), "German");
        assert_eq!("日本語".parse::<Language>().unwrap().name(), "Japanese");
        assert_eq!("한국어".parse::<Language>().unwrap().name(), "Korean");
        assert!("Klingonese".parse::<Language>().is_err());
    }

    #[test]
    fn every_label_of_the_detector_is_a_language() {
        for label in detect::LABELS.lines() {
            let language = Language::from_label(label)
                .unwrap_or_else(|| panic!("label {label} is not a language"));
            assert!(language.is_detectable());
            assert_eq!(
                language.name().parse::<Language>(),
                Ok(language),
                "{label}: {}",
                language.name()
            );
        }
    }

    #[test]
    fn a_language_the_detector_cannot_output_does_not_parse() {
        // Latin is a language, but the detector has no label for it.
        assert_eq!(
            "Latin".parse::<Language>(),
            Err(UnknownLanguage("Latin".to_string()))
        );
    }

    #[test]
    fn languages_are_enforced_only_for_a_non_english_communication_language() {
        let chinese: Language = "Chinese".parse().unwrap();
        assert!(Languages::new(chinese).enforced());
        assert!(!Languages::unset().enforced());
        assert!(!Languages::new(Language::ENGLISH).enforced());
        assert!(
            !Languages::new(Language::ENGLISH)
                .with_working(chinese)
                .enforced()
        );
    }
}
