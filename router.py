"""
Intent Router — classifies a user query as 'invoice' or 'rag'.

Optimisations:
- Uses compiled regex sets for O(1)-ish matching
- Word-boundary matching to avoid false positives
  (e.g. "note" inside "notebook" won't trigger)
- Easy to extend with new languages
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Verb patterns — intent to *create / generate*
# ---------------------------------------------------------------------------
_VERBS: list[str] = [
    # English
    "generate", "create", "make", "prepare", "build", "draft",
    # Mandarin (pinyin)
    "shengcheng", "chuangjian", "zhizuo",
    # Hindi (romanised)
    "bana", "banao", "banado", "tayar", "likho",
    # Spanish
    "generar", "crear", "hacer", "preparar",
    # French
    "generer", "créer", "creer", "faire", "preparer",
    # Arabic (romanised)
    "anshe", "insha", "awjid", "aamil",
    # Bengali (romanised)
    "toiri", "banao", "prostu",
    # Portuguese
    "gerar", "criar", "fazer", "preparar",
    # Russian (romanised)
    "sozdat", "sdelat", "podgotovit",
    # Japanese (romanised)
    "tsukuru", "sakusei",
]

# ---------------------------------------------------------------------------
# Noun patterns — the *invoice / bill* concept
# ---------------------------------------------------------------------------
_NOUNS: list[str] = [
    # English
    "invoice", "bill", "receipt",
    # Mandarin
    "fapiao", "zhangdan", "发票", "账单",
    # Hindi
    "rasid", "raseed", "bījak",
    # Spanish
    "factura", "recibo", "boleta",
    # French
    "facture", "reçu", "recu",
    # Arabic (romanised + script)
    "fatora", "wasol", "فاتورة",
    # Bengali
    "fattura", "rasid",
    # Portuguese
    "fatura", "recibo", "nota fiscal",
    # Russian (romanised)
    "schet", "kvitantsiya",
    # Japanese (romanised + script)
    "seikyusho", "ryoshusho", "請求書", "領収書",
]

# Pre-compile patterns with word boundaries for Latin-script words,
# and plain substring match for CJK / Arabic script tokens.
def _build_pattern(words: list[str]) -> re.Pattern:
    """Build a single compiled regex that matches any of the *words*."""
    parts: list[str] = []
    for w in words:
        # CJK / Arabic script — no word boundaries needed
        if re.search(r"[^\x00-\x7F]", w):
            parts.append(re.escape(w))
        else:
            parts.append(rf"\b{re.escape(w)}\b")
    return re.compile("|".join(parts), re.IGNORECASE)


_VERB_RE = _build_pattern(_VERBS)
_NOUN_RE = _build_pattern(_NOUNS)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def route(query: str) -> str:
    """Return ``'invoice'`` if the user wants to *generate* an invoice,
    otherwise ``'rag'`` for document Q&A."""
    if _VERB_RE.search(query) and _NOUN_RE.search(query):
        return "invoice"
    return "rag"