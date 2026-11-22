"""Text cleaning and input construction."""
import html
import re
import unicodedata

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def clean_text(value) -> str:
    """Normalize a raw string. Non-string values (NaN, None) become ''.

    Case and punctuation are kept on purpose: bert-base-uncased lowercases
    internally and punctuation carries useful signal for news style.
    """
    if not isinstance(value, str):
        return ""
    value = html.unescape(value)
    value = unicodedata.normalize("NFKC", value)
    value = HTML_RE.sub(" ", value)
    value = URL_RE.sub(" ", value)
    return WS_RE.sub(" ", value).strip()


def build_input_text(title, text) -> str:
    """Combine headline and body into a single model input."""
    t = clean_text(title)
    b = clean_text(text)
    if t and b:
        return f"{t}. {b}"
    return t or b
