//! Judging the language of a text's prose.
//!
//! Code blocks, code spans, URLs, paths and identifiers are removed first.
//! What is left is judged paragraph by paragraph, each by its matrix
//! language, the language of the script that carries the paragraph. A
//! paragraph is Chinese, Japanese or Korean when characters of those scripts
//! outweigh the rest, and then English or any other Latin text inside it is an
//! insertion of whatever length. Kana settle Japanese and Hangul settle
//! Korean; Han alone settles Chinese when it carries Chinese function words.
//! Those verdicts are deterministic. The paragraphs left (Han without such
//! cues, Latin, Cyrillic, Arabic and other scripts) go to a statistical
//! detector, and only when long enough to judge: it is unreliable on a short
//! phrase (FLORES-200 devtest: 88% on two to four words, 99.9% on sentences),
//! and a shorter paragraph is a heading, a label or a one-line aside.

use std::sync::Arc;

use super::Language;

/// A paragraph goes to the detector from this weight: 40 letters of a spaced
/// script, about eight words, or 20 characters of Han without cues, which
/// weigh 2 each.
const JUDGED_WEIGHT: usize = 40;

/// Characters that only Chinese uses for function words, in simplified and
/// traditional forms: Japanese does not write them in kanji.
const CHINESE_ONLY: &str = "们們这這么麼吗嗎没沒吧呢哪您咱啊嘛";
/// Frequent Chinese function words that Japanese also writes in kanji. Chinese
/// prose is dense with them, a kanji-only Japanese phrase is not.
const CHINESE_COMMON: &str = "的了是不在有我你他她它也就都和把被";
/// How many Han characters one shared function word must at most stand for
/// when no Chinese-only character is there.
const COMMON_SPAN: usize = 12;
/// How many letters of other scripts one character of Chinese, Japanese or
/// Korean holds in a paragraph's balance. Their characters carry the grammar
/// (particles, endings, function words), so a paragraph with enough of them
/// is a CJK paragraph even when an English phrase outweighs them; an English
/// paragraph with a Chinese name or term in it has far fewer.
const CJK_SPAN: usize = 8;
/// Words in a script other than Chinese, Japanese or Korean that are more than
/// a term: a paragraph of this many is not written in a CJK language.
const TERM_WORDS: usize = 3;

/// Collapses every run of whitespace to one space and trims the ends.
pub(super) fn normalize(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Judges the prose of `text` against `target`: the number, counted from 1
/// among the text's blank-line separated blocks, of the first paragraph whose
/// matrix language is another one, or `None` when there is none. `detect`
/// identifies the paragraphs the deterministic rules do not settle; it is
/// CPU work.
pub(super) fn first_paragraph_not_in(
    text: &str,
    target: Language,
    mut detect: impl FnMut(&str) -> Language,
) -> Option<usize> {
    paragraphs(text).into_iter().find_map(|paragraph| {
        let language = match Counts::of(&paragraph.text).matrix() {
            Matrix::Settled(language) => language,
            // Han alone is never a language of another script.
            Matrix::Han { .. } if !target.is_chinese_or_japanese() => {
                return Some(paragraph.number);
            }
            // Nor is a phrase of words in another script a CJK language.
            Matrix::Other { .. }
                if target.is_cjk()
                    && paragraph
                        .text
                        .split_whitespace()
                        .filter(|word| word.chars().any(char::is_alphabetic))
                        .count()
                        >= TERM_WORDS =>
            {
                return Some(paragraph.number);
            }
            Matrix::Han { weight } | Matrix::Other { weight } if weight >= JUDGED_WEIGHT => {
                detect(&paragraph.text)
            }
            Matrix::Han { .. } | Matrix::Other { .. } => return None,
        };
        (language != target).then_some(paragraph.number)
    })
}

/// What decides the matrix language of a paragraph.
#[derive(Debug, PartialEq, Eq)]
enum Matrix {
    /// A script and its function words settle the language.
    Settled(Language),
    /// Han without kana, Hangul or Chinese function words: only the detector
    /// tells Chinese from Japanese. `weight` is how much prose there is.
    Han { weight: usize },
    /// Letters of another script outweigh Chinese, Japanese and Korean; only
    /// the detector tells the language. `weight` is how much prose there is.
    Other { weight: usize },
}

/// The letters of a paragraph by script.
#[derive(Debug, Default)]
struct Counts {
    kana: usize,
    hangul: usize,
    han: usize,
    /// Characters of Han that only Chinese uses for function words.
    chinese_only: usize,
    /// Han characters that are frequent Chinese function words.
    chinese_common: usize,
    /// Letters of every other script.
    other: usize,
}

impl Counts {
    fn of(text: &str) -> Self {
        let mut counts = Self::default();
        for c in text.chars().filter(|c| c.is_alphabetic()) {
            match c {
                '\u{3040}'..='\u{30FF}' | '\u{31F0}'..='\u{31FF}' | '\u{FF66}'..='\u{FF9F}' => {
                    counts.kana += 1;
                }
                '\u{1100}'..='\u{11FF}' | '\u{3130}'..='\u{318F}' | '\u{AC00}'..='\u{D7AF}' => {
                    counts.hangul += 1;
                }
                '\u{3400}'..='\u{4DBF}'
                | '\u{4E00}'..='\u{9FFF}'
                | '\u{F900}'..='\u{FAFF}'
                | '\u{20000}'..='\u{2FA1F}' => {
                    counts.han += 1;
                    counts.chinese_only += usize::from(CHINESE_ONLY.contains(c));
                    counts.chinese_common += usize::from(CHINESE_COMMON.contains(c));
                }
                _ => counts.other += 1,
            }
        }
        counts
    }

    /// Characters of Chinese, Japanese or Korean.
    const fn cjk(&self) -> usize {
        self.kana + self.hangul + self.han
    }

    const fn matrix(&self) -> Matrix {
        if self.cjk() * CJK_SPAN < self.other {
            return Matrix::Other { weight: self.other };
        }
        if self.kana > 0 && self.kana >= self.hangul {
            return Matrix::Settled(Language::JAPANESE);
        }
        if self.hangul > 0 {
            return Matrix::Settled(Language::KOREAN);
        }
        if self.chinese_only > 0
            || (self.chinese_common >= 2 && self.chinese_common * COMMON_SPAN >= self.han)
        {
            return Matrix::Settled(Language::CHINESE);
        }
        Matrix::Han {
            weight: self.han * 2 + self.other,
        }
    }
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
#[allow(clippy::doc_markdown, reason = "FLORES-200 file names in comments")]
mod tests {
    use super::super::detect::Detector;
    use super::*;

    fn language(name: &str) -> Language {
        name.parse().unwrap()
    }

    /// The first paragraph not in `target`, the statistical cases decided by
    /// the real detector.
    fn judge_text(text: &str, target: &str) -> Option<usize> {
        let detector = Detector::load().unwrap();
        first_paragraph_not_in(text, language(target), |paragraph| {
            detector.detect(paragraph).unwrap()
        })
    }

    /// The same, for text the deterministic rules settle: the detector must
    /// not be asked.
    fn settled(text: &str, target: &str) -> Option<usize> {
        first_paragraph_not_in(text, language(target), |paragraph| {
            panic!("the detector was asked about: {paragraph}")
        })
    }

    /// zho_Hans line 12: Chinese with an English name.
    const ZH_ZMAPP: &str = "一种名为 ZMapp 的“抗体鸡尾酒”（多抗体联合）最初有望在该领域发挥作用，但正式研究表明，它在预防死亡方面的效果有些不尽人意。";
    /// zho_Hans line 13.
    const ZH_PALM: &str =
        "在 PALM 实验中，ZMapp 用作参照。也就是科学家将其作为基线，把其他三种治疗方法与之作比较。";
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

    /// A phrase of eng_Latn line 20, a longer English insertion than any term.
    const EN_PHRASE: &str = "the Camp David Accords of 1978, the normalizing of US–China relations";

    #[test]
    fn kana_hangul_and_chinese_function_words_settle_their_language_without_the_detector() {
        for (text, target) in [
            (ZH_ZMAPP, "Chinese"),
            (ZH_PALM, "Chinese"),
            (ZH_HANT, "Chinese"),
            (JA_NSA, "Japanese"),
            (KO_VACCINE, "Korean"),
        ] {
            assert_eq!(settled(text, target), None, "{target}: {text}");
        }
        // The same text against another CJK language is refused, kanji-heavy
        // Japanese as Chinese included.
        assert_eq!(settled(JA_NSA, "Chinese"), Some(1));
        assert_eq!(settled(ZH_PALM, "Japanese"), Some(1));
        assert_eq!(settled(ZH_HANT, "Korean"), Some(1));
        assert_eq!(settled(KO_VACCINE, "Japanese"), Some(1));
    }

    #[test]
    fn latin_script_paragraphs_go_to_the_detector() {
        for (text, target) in [(EN_VACCINE, "English"), (FR_VACCINE, "French")] {
            assert_eq!(judge_text(text, target), None, "{target}: {text}");
        }
        assert_eq!(judge_text(EN_VACCINE, "French"), Some(1));
        assert_eq!(judge_text(FR_VACCINE, "English"), Some(1));
        assert_eq!(judge_text(EN_VACCINE, "Chinese"), Some(1));
        assert_eq!(judge_text(ZH_HANT, "English"), Some(1));
    }

    #[test]
    fn kanji_only_japanese_without_cues_goes_to_the_detector() {
        // The articles of the 1946 constitution's renunciation of war, written
        // in kanji alone: no kana, no Hangul, no Chinese function word.
        let kanji = "日本国憲法第九条戦争放棄軍備及交戦権否認";
        assert_eq!(Counts::of(kanji).matrix(), Matrix::Han { weight: 40 });
        assert_eq!(judge_text(kanji, "Japanese"), None);
        assert_eq!(judge_text(kanji, "English"), Some(1));
    }

    #[test]
    fn english_inside_cjk_prose_is_an_insertion_of_any_length() {
        // zho_Hans line 13 with a phrase inserted, and the same for Japanese
        // and Korean.
        let chinese =
            format!("在 PALM 实验中，ZMapp 用作参照，{EN_PHRASE}。也就是科学家将其作为基线。");
        let japanese =
            format!("NSAとしてカーターを補佐し、{EN_PHRASE} など、世界問題を外交的に処理した。");
        let korean =
            format!("한 실험용 백신이 {EN_PHRASE} 에볼라 사망률을 줄일 수 있을 것처럼 보입니다.");
        assert_eq!(settled(&chinese, "Chinese"), None);
        assert_eq!(settled(&japanese, "Japanese"), None);
        assert_eq!(settled(&korean, "Korean"), None);
        // An English sentence with a few Chinese words is English.
        let english = format!("{EN_ZMAPP} It is called 抗体鸡尾酒 in Chinese.");
        assert_eq!(judge_text(&english, "English"), None);
        assert_eq!(judge_text(&english, "Chinese"), Some(1));
    }

    #[test]
    fn a_paragraph_in_another_language_is_refused_and_named() {
        let text = format!("{ZH_ZMAPP}\n\n{ZH_PALM}\n\n{EN_VACCINE}");
        assert_eq!(judge_text(&text, "Chinese"), Some(3));
        let text = format!("{EN_ZMAPP}\n\n{ZH_ZMAPP}");
        assert_eq!(judge_text(&text, "English"), Some(2));
        // A Japanese paragraph in a Chinese text, though each reads as a
        // language of its own.
        let text = format!("{ZH_PALM}\n\n{JA_NSA}");
        assert_eq!(judge_text(&text, "Chinese"), Some(2));
    }

    #[test]
    fn short_paragraphs_of_the_detector_are_not_judged() {
        // The detector is unreliable on a short French line, so it is not asked.
        let fr_short = "Il l'a dit hier.";
        assert!(Counts::of(fr_short).other < JUDGED_WEIGHT);
        let text = format!("{EN_VACCINE}\n\n{fr_short}\n\n{EN_ZMAPP}");
        assert_eq!(judge_text(&text, "English"), None);
        // A phrase of Latin words is never Chinese, whatever its length.
        let text = format!("{ZH_ZMAPP}\n\n{EN_SHORT}\n\n{ZH_PALM}");
        assert_eq!(settled(&text, "Chinese"), Some(2));
        // A term or two in a paragraph of their own are not judged.
        assert_eq!(settled("Summary\n\nPALM trial", "Chinese"), None);
    }

    #[test]
    fn code_identifiers_paths_and_quotes_are_not_prose() {
        let text = format!(
            "{ZH_PALM}\n\n```rust\n{EN_VACCINE}\n\nfn main() {{}}\n```\n\n请检查 `{EN_ZMAPP}` 和 spawn_blocking，见 src/agent/mod.rs 与 https://example.com/a。"
        );
        assert_eq!(settled(&text, "Chinese"), None);
        assert_eq!(
            settled("`cargo test` https://example.com 42", "Chinese"),
            None
        );
    }

    #[test]
    fn code_between_paragraphs_still_counts_as_one_block_for_numbering() {
        let text = format!("{ZH_PALM}\n\n```\ncode\n\nmore\n```\n\n{EN_VACCINE}");
        assert_eq!(judge_text(&text, "Chinese"), Some(3));
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
