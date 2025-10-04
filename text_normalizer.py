"""Utility helpers to normalize textual content scraped with mojibake artifacts."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, TYPE_CHECKING

try:
    from ftfy import fix_text  # type: ignore
except Exception:  # pragma: no cover - ftfy is optional at runtime
    fix_text = None  # pyright: ignore[reportPrivateImportUsage]

if TYPE_CHECKING:  # pragma: no cover - only for static typing
    from langchain.schema import Document


# Characters that frequently appear in the dumped text because of encoding
# round-trips (UTF-8 bytes interpreted as Latin-1, CP-1252 artifacts, etc.).
# We translate them into their intended Portuguese characters.
_CHAR_TRANSLATION = str.maketrans({
    "§": "ç",
    "£": "ã",
    "¡": "á",
    "©": "é",
    "³": "ó",
    "µ": "õ",
    "¢": "â",
    "\u00a0": " ",  # non-breaking space -> plain space
    "\u00ad": None,   # soft hyphen -> remove
})


# Regex helpers for context-aware replacements that cannot be solved with a
# simple character table (e.g., ordinal indicators and feminine abbreviations).
_INTERNAL_FEMININE_A = re.compile(r"(?<=\w)ª(?=\w)")
_ORDINAL_LOWER = re.compile(r"(?<![0-9nN])º(?=[a-záéíóúãõêô])")
_ORDINAL_UPPER = re.compile(r"(?<![0-9nN])º(?=[A-ZÁÉÍÓÚÃÕÊÔ])")


def normalize_text(text: str) -> str:
    """Normalise Portuguese text, fixing mojibake and stray control chars."""

    if not text:
        return ""

    cleaned = text

    # First pass: rely on ftfy (when available) to undo classic UTF-8/Latin-1
    # swap issues such as "Ã§", "â\x80\x93", etc.
    if fix_text is not None:
        cleaned = fix_text(cleaned, normalization="NFC")
    else:  # graceful fallback preserves the original text on failure
        try:
            cleaned = cleaned.encode("latin-1").decode("utf-8")
        except UnicodeDecodeError:
            cleaned = unicodedata.normalize("NFC", cleaned)

    # Second pass: apply bespoke replacements observed in the corpus.
    cleaned = cleaned.translate(_CHAR_TRANSLATION)
    cleaned = _INTERNAL_FEMININE_A.sub("ê", cleaned)
    cleaned = _ORDINAL_LOWER.sub("ú", cleaned)
    cleaned = _ORDINAL_UPPER.sub("Ú", cleaned)

    # Specific fixes that the generic passes cannot capture.
    cleaned = cleaned.replace("Ant´nio", "Antônio")

    # Normalise again to NFC to keep composed characters.
    cleaned = unicodedata.normalize("NFC", cleaned)

    return cleaned


def normalize_documents(docs: Iterable["Document"] | None) -> list["Document"]:
    """Apply ``normalize_text`` to every document in-place."""

    if not docs:
        return []

    normalised: list["Document"] = []
    for doc in docs:
        if doc is None:
            continue
        original = getattr(doc, "page_content", "") or ""
        fixed = normalize_text(original)
        if fixed != original:
            doc.page_content = fixed
        normalised.append(doc)
    return normalised

