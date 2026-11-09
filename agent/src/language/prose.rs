//! Judging the language of a text's prose.
//!
//! Code blocks, code spans, URLs, paths and identifiers are removed first.
//! The prose that is left must be detected as the expected language as a
//! whole, and so must every paragraph long enough to judge. A shorter
//! paragraph is a heading, a label or a one-line aside, and the detector is
//! unreliable on it (FLORES-200 devtest: 88% on phrases of two to four words,
//! 99.9% on sentences), so it is not judged; a short insertion inside a
//! judged paragraph, such as an English term in Chinese prose, is carried by
//! the paragraph's language.

use std::sync::Arc;

use super::Language;
use super::detect::Detector;

/// A paragraph is judged on its own from this weight: 40 letters of a spaced
/// script, about eight words, or 20 characters of Chinese, Japanese or
/// Korean, which weigh 2 each.
const JUDGED_WEIGHT: usize = 40;

/// Collapses every run of whitespace to one space and trims the ends.
pub(super) fn normalize(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// How a text fared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Judgement {
    /// The prose is in the expected language, or there is none.
    InLanguage,
    /// It is not. `paragraph` is the number, counted from 1 in the text, of
    /// the first judged paragraph in another language; `None` when each is in
    /// the language but the prose as a whole is not.
    NotIn { paragraph: Option<usize> },
}

/// Judges the prose of `text` against `target`. This runs the detector, which
/// is CPU work.
///
/// # Panics
///
/// When the loaded detector fails to run on a text.
pub(super) fn judge(detector: &Detector, text: &str, target: Language) -> Judgement {
    let detect = |text: &str| {
        detector
            .detect(text)
            .expect("the language detector runs on any text")
    };
    let paragraphs = paragraphs(text);
    let mut judged = 0;
    for paragraph in &paragraphs {
        if weight(&paragraph.text) < JUDGED_WEIGHT {
            continue;
        }
        judged += 1;
        if detect(&paragraph.text) != target {
            return Judgement::NotIn {
                paragraph: Some(paragraph.number),
            };
        }
    }
    if paragraphs.is_empty() || (paragraphs.len() == 1 && judged == 1) {
        // No prose, or the one paragraph was the whole.
        return Judgement::InLanguage;
    }
    let whole = paragraphs
        .iter()
        .map(|paragraph| paragraph.text.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    if detect(&whole) == target {
        Judgement::InLanguage
    } else {
        Judgement::NotIn { paragraph: None }
    }
}

/// How much prose `text` holds: one for each letter, two for each Chinese,
/// Japanese or Korean character.
fn weight(text: &str) -> usize {
    text.chars()
        .filter(|c| c.is_alphabetic())
        .map(|c| if is_wide(c) { 2 } else { 1 })
        .sum()
}

const fn is_wide(c: char) -> bool {
    matches!(
        c,
        '\u{3040}'..='\u{30FF}'
            | '\u{3400}'..='\u{4DBF}'
            | '\u{4E00}'..='\u{9FFF}'
            | '\u{AC00}'..='\u{D7AF}'
    )
}

/// Removes the quotes of `text` that match, after whitespace normalization,
/// something in `corpus`. A quote is a span in double quotes (straight,
/// curly, corner or guillemets) or a markdown blockquote line.
pub(super) fn without_quotes(text: &str, corpus: &[Arc<str>]) -> String {
    let matches = |quoted: &str| {
        let quoted = normalize(quoted);
        !quoted.is_empty() && corpus.iter().any(|message| message.contains(&quoted))
    };
    let mut unquoted = String::with_capacity(text.len());
    for line in text.lines() {
        if let Some(quoted) = line.trim_start().strip_prefix('>')
            && matches(quoted)
        {
            continue;
        }
        unquoted.push_str(line);
        unquoted.push('\n');
    }
    let mut kept = String::with_capacity(unquoted.len());
    let mut chars = unquoted.chars();
    while let Some(c) = chars.next() {
        let Some(close) = closing_quote(c) else {
            kept.push(c);
            continue;
        };
        let mut quoted = String::new();
        let mut closed = false;
        for inner in chars.by_ref() {
            if inner == close {
                closed = true;
                break;
            }
            quoted.push(inner);
        }
        if closed && matches(&quoted) {
            kept.push(' ');
        } else {
            kept.push(c);
            kept.push_str(&quoted);
            if closed {
                kept.push(close);
            }
        }
    }
    kept
}

const fn closing_quote(open: char) -> Option<char> {
    match open {
        '"' => Some('"'),
        '\u{201c}' => Some('\u{201d}'),
        '\u{300c}' => Some('\u{300d}'),
        '\u{300e}' => Some('\u{300f}'),
        '\u{ab}' => Some('\u{bb}'),
        _ => None,
    }
}

/// A blank-line separated block of the text, reduced to its prose.
struct Paragraph {
    /// Its place among the blocks of the text, counted from 1.
    number: usize,
    text: String,
}

/// The paragraphs of `text` that hold prose, with code, URLs, paths and
/// identifiers removed.
fn paragraphs(text: &str) -> Vec<Paragraph> {
    let mut paragraphs = Vec::new();
    let mut number = 0;
    let mut current = String::new();
    let mut fenced = false;
    let mut after_blank = true;
    let mut finish = |number: usize, current: &mut String| {
        if current.chars().any(char::is_alphabetic) {
            paragraphs.push(Paragraph {
                number,
                text: std::mem::take(current),
            });
        } else {
            current.clear();
        }
    };
    for line in text.lines() {
        let trimmed = line.trim_start();
        if !fenced && trimmed.is_empty() {
            finish(number, &mut current);
            after_blank = true;
            continue;
        }
        if after_blank {
            number += 1;
            after_blank = false;
        }
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            fenced = !fenced;
            continue;
        }
        if fenced {
            continue;
        }
        for (index, part) in line.split('`').enumerate() {
            if index % 2 == 0 {
                push_words(part, &mut current);
            }
        }
        current.push('\n');
    }
    finish(number, &mut current);
    paragraphs
}

fn push_words(text: &str, out: &mut String) {
    for token in text.split_whitespace() {
        if token.contains("://") || token.starts_with("www.") {
            out.push(' ');
            continue;
        }
        let mut chunk = String::new();
        let mut chunk_is_latin = true;
        for c in token.chars() {
            let latin = c.is_ascii() || (c as u32) < 0x250;
            if latin != chunk_is_latin {
                flush(&chunk, chunk_is_latin, out);
                chunk.clear();
                chunk_is_latin = latin;
            }
            chunk.push(c);
        }
        flush(&chunk, chunk_is_latin, out);
        out.push(' ');
    }
}

fn flush(chunk: &str, latin: bool, out: &mut String) {
    if chunk.is_empty() {
        return;
    }
    if latin && is_identifier(chunk) {
        out.push(' ');
    } else {
        out.push_str(chunk);
    }
}

/// Whether a run of Latin-range characters is a path, an identifier, a flag
/// or an acronym rather than a word of prose.
fn is_identifier(chunk: &str) -> bool {
    let core = chunk.trim_matches(|c: char| {
        matches!(
            c,
            '(' | ')'
                | '['
                | ']'
                | '{'
                | '}'
                | '<'
                | '>'
                | ','
                | ';'
                | '!'
                | '?'
                | '"'
                | '\''
                | '*'
                | ':'
                | '.'
        )
    });
    if core.is_empty() {
        return false;
    }
    if core.starts_with('-') && core.len() > 1 {
        return true;
    }
    if core.contains(['_', '/', '\\', '~', '@', '#', '=', '.', '$'])
        || core.contains("::")
        || core.contains(|c: char| c.is_ascii_digit())
    {
        return true;
    }
    let mut previous: Option<char> = None;
    for c in core.chars() {
        if previous.is_some_and(char::is_lowercase) && c.is_uppercase() {
            return true;
        }
        previous = Some(c);
    }
    core.chars().count() >= 2 && core.chars().all(char::is_uppercase)
}

// The test sentences are lines of FLORES-200 devtest (NLLB Team, Meta AI;
// CC BY-SA 4.0), `https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz`,
// named by language file and 1-based line.
#[cfg(test)]
mod tests {
    use super::*;

    fn language(name: &str) -> Language {
        name.parse().unwrap()
    }

    fn judge_text(text: &str, target: &str) -> Judgement {
        judge(&Detector::load().unwrap(), text, language(target))
    }

    /// zho_Hans line 12: Chinese with an English name.
    const ZH_ZMAPP: &str = "一种名为 ZMapp 的“抗体鸡尾酒”（多抗体联合）最初有望在该领域发挥作用，但正式研究表明，它在预防死亡方面的效果有些不尽人意。";
    /// zho_Hans line 13.
    const ZH_PALM: &str = "在 PALM 实验中，ZMapp 用作参照。也就是科学家将其作为基线，把其他三种治疗方法与之作比较。";
    /// zho_Hant line 11.
    const ZH_HANT: &str = "儘管有一種實驗性的疫苗顯示能降低伊波拉病毒的致死率，但到目前為止，沒有藥物明確證實適合治療已受感染的人。";
    /// eng_Latn line 11.
    const EN_VACCINE: &str = "While one experimental vaccine appears able to reduce Ebola mortality, up until now, no drugs have been clearly demonstrated suitable for treating existing infection.";
    /// eng_Latn line 12.
    const EN_ZMAPP: &str = "One antibody cocktail, ZMapp, initially showed promise in the field, but formal studies indicated it had less benefit than sought in preventing death.";
    /// eng_Latn line 7, too short to be judged.
    const EN_SHORT: &str = "He built a WiFi door bell, he said.";
    /// jpn_Jpan line 20: mostly kanji, with kana.
    const JA_NSA: &str = "NSAとしてカーターを補佐し、1978年のキャンプ・デイビッド合意、1970年代後半の米中関係の正常化、1979年のイラン人質事件につながったイラン革命、1979年のソビエトのアフガニスタン侵攻など、世界問題を外交的に処理した。";
    /// kor_Hang line 11.
    const KO_VACCINE: &str = "한 실험용 백신이 에볼라 사망률을 줄일 수 있을 것처럼 보이지만, 지금까지 어떤 약도 기존의 감염을 치료하는데 적합하다고 명확하게 입증되지 않았습니다.";
    /// fra_Latn line 11.
    const FR_VACCINE: &str = "Même si un vaccin expérimental semble jusqu'ici capable de réduire le taux de mortalité d'Ebola, aucun remède n'a encore été suffisamment efficace pour traiter l'infection existante.";

    #[test]
    fn chinese_japanese_and_korean_prose_is_judged_in_its_own_language() {
        for (text, target) in [
            (ZH_ZMAPP, "Chinese"),
            (ZH_PALM, "Chinese"),
            (ZH_HANT, "Chinese"),
            (JA_NSA, "Japanese"),
            (KO_VACCINE, "Korean"),
            (EN_VACCINE, "English"),
            (FR_VACCINE, "French"),
        ] {
            assert_eq!(
                judge_text(text, target),
                Judgement::InLanguage,
                "{target}: {text}"
            );
        }
    }

    #[test]
    fn kanji_heavy_japanese_is_japanese_and_not_chinese() {
        assert_eq!(
            judge_text(JA_NSA, "Chinese"),
            Judgement::NotIn { paragraph: Some(1) }
        );
    }

    #[test]
    fn cjk_prose_with_english_terms_and_a_longer_english_phrase_is_accepted() {
        // zho_Hans line 13 with a phrase of eng_Latn line 20 inserted.
        let mixed = "在 PALM 实验中，ZMapp 用作参照 the Camp David Accords, 1978; normalizing US–China relations。也就是科学家将其作为基线，把其他三种治疗方法与之作比较。";
        assert_eq!(judge_text(mixed, "Chinese"), Judgement::InLanguage);
    }

    #[test]
    fn a_paragraph_in_another_language_is_refused_and_named() {
        let text = format!("{ZH_ZMAPP}\n\n{ZH_PALM}\n\n{EN_VACCINE}");
        assert_eq!(
            judge_text(&text, "Chinese"),
            Judgement::NotIn { paragraph: Some(3) }
        );
        let text = format!("{EN_ZMAPP}\n\n{ZH_ZMAPP}");
        assert_eq!(
            judge_text(&text, "English"),
            Judgement::NotIn { paragraph: Some(2) }
        );
    }

    #[test]
    fn a_wholly_wrong_language_is_refused() {
        assert_eq!(
            judge_text(EN_VACCINE, "Chinese"),
            Judgement::NotIn { paragraph: Some(1) }
        );
        assert_eq!(
            judge_text(ZH_HANT, "English"),
            Judgement::NotIn { paragraph: Some(1) }
        );
        assert_eq!(
            judge_text(KO_VACCINE, "Japanese"),
            Judgement::NotIn { paragraph: Some(1) }
        );
    }

    #[test]
    fn short_paragraphs_are_not_judged() {
        let text = format!("{ZH_ZMAPP}\n\n{EN_SHORT}\n\n{ZH_PALM}");
        assert_eq!(judge_text(&text, "Chinese"), Judgement::InLanguage);
        assert!(weight(EN_SHORT) < JUDGED_WEIGHT);
        // A short sentence alone is not judged either.
        assert_eq!(judge_text(EN_SHORT, "Chinese"), Judgement::InLanguage);
    }

    #[test]
    fn code_identifiers_paths_and_quotes_are_not_prose() {
        let text = format!(
            "{ZH_PALM}\n\n```rust\n{EN_VACCINE}\n\nfn main() {{}}\n```\n\n请检查 `{EN_ZMAPP}` 和 spawn_blocking，见 src/agent/mod.rs 与 https://example.com/a。"
        );
        assert_eq!(judge_text(&text, "Chinese"), Judgement::InLanguage);
        assert_eq!(
            judge_text("`cargo test` https://example.com 42", "Chinese"),
            Judgement::InLanguage
        );
    }

    #[test]
    fn code_between_paragraphs_still_counts_as_one_block_for_numbering() {
        let text = format!("{ZH_PALM}\n\n```\ncode\n\nmore\n```\n\n{EN_VACCINE}");
        assert_eq!(
            judge_text(&text, "Chinese"),
            Judgement::NotIn { paragraph: Some(3) }
        );
    }

    #[test]
    fn matching_quotes_are_removed_and_others_kept() {
        let corpus = [Arc::<str>::from("请把数据库换成 PostgreSQL，不要用 MySQL")];
        let text = "User said \u{201c}请把数据库换成   PostgreSQL，不要用\nMySQL\u{201d}.";
        assert!(!without_quotes(text, &corpus).contains('库'));
        let other = "User said \u{201c}他们想要完全不同的方案\u{201d}.";
        assert!(without_quotes(other, &corpus).contains("方案"));
    }
}
