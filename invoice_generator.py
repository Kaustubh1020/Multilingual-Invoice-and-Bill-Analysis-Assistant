"""
Invoice Generator — produces clean, structured invoices via the LLM.

Optimisations:
- Dedicated system prompts for consistent role behavior
- Tighter prompt engineering to prevent hallucinated numbers
- Currency symbol map extended
- Temperature set low for deterministic output
"""

from __future__ import annotations

from llm import ask_llm

# ---------------------------------------------------------------------------
# Currency symbols
# ---------------------------------------------------------------------------
CURRENCY_SYMBOLS: dict[str, str] = {
    "USD": "$",
    "INR": "₹",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "CNY": "¥",
    "KRW": "₩",
    "RUB": "₽",
    "SAR": "﷼",
    "AED": "د.إ",
    "BDT": "৳",
    "BRL": "R$",
    "CAD": "CA$",
    "AUD": "A$",
}

_SYSTEM_PROMPT = (
    "You are a professional, multilingual invoice generator. "
    "You produce clean, well-formatted invoices and NEVER invent or change numbers. "
    "Use exact values from the data provided."
)


# ---------------------------------------------------------------------------
# Free-text invoice (from a chat query)
# ---------------------------------------------------------------------------
def generate_invoice(query: str) -> str:
    """Generate an invoice purely from a natural-language user request."""
    prompt = f"""\
User request: {query}

Generate a clean, professional invoice from the request above.

Rules:
- Detect the language of the request automatically and respond in that language.
- Include columns: Item, Quantity, Unit Price, Total.
- Add a summary section with Subtotal and Grand Total.
- Format as a readable markdown table.
- Output ONLY the invoice — no explanations.
"""
    return ask_llm(prompt, system=_SYSTEM_PROMPT, temperature=0.2)


# ---------------------------------------------------------------------------
# Excel-data invoice
# ---------------------------------------------------------------------------
def generate_invoice_from_excel(
    excel_data: list[dict],
    language: str = "English",
    src_currency: str | None = None,
    tgt_currency: str | None = None,
    rate: float | None = None,
) -> str:
    """Generate an invoice from pre-processed Excel row dicts."""

    symbol = CURRENCY_SYMBOLS.get(tgt_currency, tgt_currency or "")

    conversion_note = ""
    if src_currency and tgt_currency and rate:
        conversion_note = (
            f"CRITICAL: All monetary values below have ALREADY been converted "
            f"from {src_currency} to {tgt_currency} at 1 {src_currency} = {rate} {tgt_currency}. "
            f"DO NOT modify any numbers. Display them exactly as given."
        )

    prompt = f"""\
Data (line items + optional summary object at the end):
{excel_data}

{conversion_note if conversion_note else 'Amounts are in their original currency.'}

Column mapping:
  • item        → product / service name
  • description → optional extra detail
  • quantity    → number of units
  • unit_price  → price per unit
  • total       → line total (discounts already applied — do NOT recalculate)
  • tax         → tax amount per line (if present)
  • discount    → discount already applied (if present)

Summary handling:
If a summary dict exists (keys: subtotal, tax, discount, grand_total):
  → Use ONLY the values present; do NOT fabricate missing keys.
  → Translate labels into {language}.

If NO summary dict exists:
  → subtotal = sum of all "total" values.
  → total_tax = sum of "tax" column (0 if absent).
  → grand_total = subtotal + total_tax.

Output language : {language}
Currency symbol : {symbol}

Produce a clean markdown invoice:
1. A line-item table (Item | Qty | Unit Price | Total).
2. A summary block (Subtotal, Tax, Grand Total — only fields that exist).

Output ONLY the invoice.
"""
    return ask_llm(prompt, system=_SYSTEM_PROMPT, temperature=0.15, max_tokens=3000)