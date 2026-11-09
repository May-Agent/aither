"""Constants of the language-detector Core ML conversion."""

# Sequence-length buckets, one Core ML function each. A FLORES-200 sentence is
# 20 to 80 tokens; the Rust side truncates to the largest bucket.
BUCKETS = (32, 64, 128)
FUNCTION_PREFIX = "len_"
DEFAULT_FUNCTION = "len_32"

# The additive attention-mask value of a padding position; finite so fp16 holds it.
NEG = -1e4

# FLORES-200 devtest files checked, the languages of the guard's tests, and the
# number of lines taken from the top of each.
FLORES_LANGUAGES = (
    "zho_Hans", "zho_Hant", "jpn_Jpan", "kor_Hang", "eng_Latn", "fra_Latn",
    "deu_Latn", "spa_Latn", "ita_Latn", "por_Latn", "nld_Latn", "rus_Cyrl",
    "arb_Arab", "pes_Arab", "urd_Arab",
)
FLORES_LINES = 100

# The Core ML model must pick the same language as the ONNX model the other
# platforms run, and as the fp32 weights it was converted from, on at least this
# share of the sentences. The guard needs the same top label, not equal logits.
MIN_AGREEMENT = 0.99

# Share of the operations of each function placed on the Neural Engine when the
# model may use CPU and Neural Engine.
MIN_ANE_SHARE = 0.9

SOURCE_REPO = "alexneakameni/language_detection"
SOURCE_REVISION = "41e397970e2e598b90837a3e2000d885e59414f3"
SHORT_REVISION = SOURCE_REVISION[:8]
