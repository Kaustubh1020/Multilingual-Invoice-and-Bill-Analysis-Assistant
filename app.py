"""
AI Invoice & Bill Assistant — Streamlit Application
Premium UI with dark theme, chat interface, and robust file processing.
"""

import copy
import streamlit as st
import pdfplumber
import pandas as pd
import requests
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

from router import route
from invoice_generator import generate_invoice, generate_invoice_from_excel
from rag_pipeline import (
    add_documents, add_raw_text, retrieve, is_count_query, count_bills,
    bill_registry, documents,
)
from llm import ask_llm

# ════════════════════════════════════════════════════════════════════
# PAGE CONFIG & CUSTOM CSS
# ════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Invoice Assistant",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Header ── */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.8rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(102,126,234,.25);
}
.main-header h1 {
    color: #fff; margin: 0; font-size: 2rem; font-weight: 700;
    letter-spacing: -0.5px;
}
.main-header p { color: rgba(255,255,255,.85); margin: .3rem 0 0; font-size: .95rem; }

/* ── Chat bubbles ── */
.chat-user, .chat-bot {
    padding: 1rem 1.25rem;
    border-radius: 14px;
    margin-bottom: .75rem;
    max-width: 85%;
    line-height: 1.55;
    font-size: .93rem;
    animation: fadeSlide .35s ease;
}
.chat-user {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: #fff; margin-left: auto; border-bottom-right-radius: 4px;
}
.chat-bot {
    background: #f0f2f6; color: #1a1a2e;
    border-bottom-left-radius: 4px;
}
@keyframes fadeSlide {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}
section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

/* ── File cards ── */
.file-card {
    background: rgba(255,255,255,.08);
    border: 1px solid rgba(255,255,255,.12);
    border-radius: 10px;
    padding: .7rem 1rem;
    margin-bottom: .5rem;
    backdrop-filter: blur(6px);
    transition: transform .2s, box-shadow .2s;
}
.file-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(102,126,234,.3);
}

/* ── Stat pill ── */
.stat-pill {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: #fff; text-align: center;
    padding: .9rem; border-radius: 12px;
    margin-bottom: 1rem;
    box-shadow: 0 4px 16px rgba(102,126,234,.3);
}
.stat-pill .num { font-size: 1.8rem; font-weight: 700; }
.stat-pill .lbl { font-size: .8rem; opacity: .85; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    padding: .55rem 1.5rem !important;
    transition: transform .2s, box-shadow .2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102,126,234,.4) !important;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# SESSION STATE DEFAULTS
# ════════════════════════════════════════════════════════════════════
_DEFAULTS = {
    "chat": [],
    "processed_files": [],
    "excel_text": None,
    "excel_filename": None,
    "column_cache": {},
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════
def get_exchange_rate(src: str, tgt: str) -> float:
    if src == tgt:
        return 1.0
    try:
        r = requests.get(f"https://api.frankfurter.app/latest?from={src}&to={tgt}", timeout=5)
        return r.json()["rates"][tgt]
    except Exception as e:
        st.warning(f"⚠️ Currency conversion failed: {e}. Using 1:1.")
        return 1.0


def get_language(query: str) -> str:
    try:
        code = detect(query)
        return {
            "en": "English", "es": "Spanish", "hi": "Hindi",
            "fr": "French", "ar": "Arabic", "bn": "Bengali",
            "pt": "Portuguese", "ru": "Russian", "ja": "Japanese",
            "zh-cn": "Chinese (Mandarin)", "zh-tw": "Chinese (Traditional)",
        }.get(code, "English")
    except Exception:
        return "English"


# ════════════════════════════════════════════════════════════════════
# SMART COLUMN MAPPING
# ════════════════════════════════════════════════════════════════════
_RULE_MAP = {
    "item": ["item", "product", "name"],
    "description": ["desc", "detail", "info"],
    "quantity": ["qty", "quantity", "units", "pcs", "nos", "count"],
    "unit_price": ["price", "rate", "cost", "unit price", "unit_price"],
    "total": ["total", "amount", "value", "fees", "charge"],
    "tax": ["tax", "gst", "vat"],
    "discount": ["discount", "disc"],
}


def rule_map(col: str) -> str:
    cl = col.lower()
    for target, keywords in _RULE_MAP.items():
        if any(k in cl for k in keywords):
            return target
    return col


def llm_map(col: str) -> str:
    if col in st.session_state.column_cache:
        return st.session_state.column_cache[col]
    prompt = (
        "Classify this Excel column name into exactly one label:\n"
        "item, description, quantity, unit_price, total, tax, discount\n\n"
        f"Column name: {col}\n\nReturn ONLY one word."
    )
    result = ask_llm(prompt, temperature=0.0, max_tokens=16).strip().lower()
    valid = {"item", "description", "quantity", "unit_price", "total", "tax", "discount"}
    if result not in valid:
        result = col
    st.session_state.column_cache[col] = result
    return result


def smart_column_mapping(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for col in df.columns:
        mapped = rule_map(col)
        mapping[col] = llm_map(col) if mapped == col else mapped
    return df.rename(columns=mapping)


# ════════════════════════════════════════════════════════════════════
# MULTILINGUAL HEADER KEYWORDS (for Excel header-row detection)
# ════════════════════════════════════════════════════════════════════
MULTILINGUAL_KEYWORDS = {
    "item": [
        "item", "product", "name", "article",
        "artículo", "producto", "nombre", "produit", "nom",
        "المادة", "المنتج", "товар", "продукт", "आइटम", "উত্পাদ",
        "商品", "名前", "项目", "产品",
    ],
    "description": [
        "desc", "description", "detail", "info",
        "descripción", "détail", "وصف", "описание", "विवरण", "বর্ণনা",
        "説明", "描述",
    ],
    "quantity": [
        "qty", "quantity", "units", "pcs", "nos", "count",
        "cantidad", "quantité", "كمية", "количество", "मात्रा", "পরিমাণ",
        "数量", "個数",
    ],
    "unit_price": [
        "price", "rate", "cost", "unit price", "unit_price",
        "precio", "prix", "سعر", "цена", "कीमत", "মূল্য",
        "価格", "单价",
    ],
    "total": [
        "total", "amount", "value", "fees", "charge",
        "monto", "montant", "المجموع", "итого", "कुल", "মোট",
        "合計", "合计",
    ],
    "tax": [
        "tax", "gst", "vat", "igst", "cgst", "sgst",
        "impuesto", "taxe", "tva", "ضريبة", "налог", "कर", "কর",
        "税金", "税",
    ],
    "discount": [
        "discount", "disc", "rebate",
        "descuento", "remise", "خصم", "скидка", "छूट", "ছাড়",
        "割引", "折扣",
    ],
}
ALL_KW = [w for group in MULTILINGUAL_KEYWORDS.values() for w in group]


# ════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🧾 AI Invoice Assistant")
    st.markdown("---")

    # Stats
    n = len(st.session_state.processed_files)
    st.markdown(
        f'<div class="stat-pill"><div class="num">{n}</div>'
        f'<div class="lbl">Document{"s" if n != 1 else ""} Uploaded</div></div>',
        unsafe_allow_html=True,
    )

    # File list
    if st.session_state.processed_files:
        st.markdown("#### 📂 Uploaded Files")
        for name in st.session_state.processed_files:
            icon = "📊" if name.endswith((".xlsx", ".xls", ".csv")) else "📄"
            st.markdown(f'<div class="file-card">{icon} {name}</div>', unsafe_allow_html=True)
    else:
        st.info("Upload a document to get started.")

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat = []
        st.rerun()


# ════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="main-header">'
    '<h1>🧾 AI Invoice & Bill Assistant</h1>'
    '<p>Upload bills (PDF / TXT / Excel) • Ask questions • Generate invoices in 10+ languages</p>'
    '</div>',
    unsafe_allow_html=True,
)


# ════════════════════════════════════════════════════════════════════
# FILE UPLOAD
# ════════════════════════════════════════════════════════════════════
uploaded_file = st.file_uploader(
    "Upload PDF, TXT or Excel",
    type=["pdf", "txt", "xlsx", "xls", "csv"],
    label_visibility="collapsed",
)

if uploaded_file and uploaded_file.name not in st.session_state.processed_files:
    ext = uploaded_file.name.rsplit(".", 1)[-1].lower()

    # ── Excel / CSV ──────────────────────────────────────────────
    if ext in ("xlsx", "xls", "csv"):
        with st.spinner("📊 Processing spreadsheet…"):
            # 1. Read raw to detect header row
            df_raw = (
                pd.read_csv(uploaded_file, header=None)
                if ext == "csv"
                else pd.read_excel(uploaded_file, header=None)
            )

            best_row, best_score = 0, 0
            for i, row in df_raw.iterrows():
                row_text = " ".join(str(c).lower() for c in row if pd.notna(c))
                score = sum(1 for kw in ALL_KW if kw in row_text)
                if score > best_score:
                    best_score, best_row = score, i

            data_start = best_row if best_score >= 2 else 0

            # 2. Re-read with detected header
            uploaded_file.seek(0)
            df = (
                pd.read_csv(uploaded_file, skiprows=data_start)
                if ext == "csv"
                else pd.read_excel(uploaded_file, skiprows=data_start)
            )
            df = df.dropna(how="all").dropna(axis=1, how="all")

            st.subheader("📊 Data Preview")
            st.dataframe(df, use_container_width=True)

            # Normalise columns
            df.columns = [str(c).lower().strip() for c in df.columns]

            # Drop serial-number columns
            serial_kw = {"sr", "sr no", "s.no", "sno", "serial", "#", "no.", "index", "sl no", "sl.no"}
            df = df[[c for c in df.columns if c.strip() not in serial_kw]]

            df = smart_column_mapping(df)
            df = df.loc[:, ~df.columns.duplicated()]

            # Clean currency symbols & coerce numerics
            CURR_RE = r"[₹$€£¥₩₽﷼৳]"
            for col in df.select_dtypes(include="object").columns:
                cleaned = (
                    df[col].astype(str)
                    .str.replace(CURR_RE, "", regex=True)
                    .str.replace(",", "", regex=False)
                    .str.strip()
                )
                coerced = pd.to_numeric(cleaned, errors="coerce")
                if coerced.notna().sum() > 0:
                    df[col] = coerced

            # Recalculate total if missing
            if "total" not in df.columns and "quantity" in df.columns and "unit_price" in df.columns:
                if "discount" in df.columns:
                    df["total"] = df["quantity"] * df["unit_price"] * (1 - df["discount"] / 100)
                else:
                    df["total"] = df["quantity"] * df["unit_price"]
            if "total" in df.columns:
                df["total"] = df["total"].round(2)

            excel_text = df.to_dict(orient="records")

            # Detect summary rows below data
            summary_kw = {
                "subtotal": ["subtotal", "sub total", "sub-total", "小計", "小计"],
                "tax": ["tax", "gst", "vat", "impuesto", "taxe", "ضريبة", "税金"],
                "discount": ["discount", "disc", "descuento", "remise", "خصم", "割引"],
                "grand_total": ["grand total", "total", "invoice total", "合計", "总计"],
            }
            summary = {}
            for i in range(data_start + 1 + len(df), len(df_raw)):
                row = df_raw.iloc[i]
                row_text = " ".join(str(c).lower() for c in row if pd.notna(c))
                if not row_text:
                    continue
                for key, kws in summary_kw.items():
                    if any(kw in row_text for kw in kws):
                        for cell in row:
                            if pd.notna(cell):
                                try:
                                    summary[key] = float(str(cell).replace(",", ""))
                                    break
                                except ValueError:
                                    continue
                        break
            if summary:
                excel_text.append({"summary": summary})

            st.session_state.excel_text = excel_text
            st.session_state.excel_filename = uploaded_file.name
            st.session_state.processed_files.append(uploaded_file.name)
            st.success(f"✅ **{uploaded_file.name}** processed — {len(df)} rows detected!")

    # ── PDF ──────────────────────────────────────────────────────
    elif ext == "pdf":
        with st.spinner("📄 Extracting PDF text…"):
            text = ""
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            if not text.strip():
                st.error("Could not extract text from this PDF.")
            else:
                add_raw_text(text, doc_name=uploaded_file.name)
                st.session_state.processed_files.append(uploaded_file.name)
                st.success(f"✅ **{uploaded_file.name}** processed!")
                st.rerun()

    # ── TXT ──────────────────────────────────────────────────────
    elif ext == "txt":
        with st.spinner("📝 Reading text file…"):
            text = uploaded_file.read().decode()
            add_raw_text(text, doc_name=uploaded_file.name)
            st.session_state.processed_files.append(uploaded_file.name)
            st.success(f"✅ **{uploaded_file.name}** processed!")
            st.rerun()


# ════════════════════════════════════════════════════════════════════
# EXCEL INVOICE GENERATOR PANEL
# ════════════════════════════════════════════════════════════════════
if st.session_state.excel_text:
    st.markdown("---")
    st.subheader(f"🧾 Generate Invoice from {st.session_state.excel_filename}")

    col1, col2 = st.columns(2)
    with col1:
        selected_lang = st.selectbox(
            "🌐 Invoice Language",
            ["Select a language…", "English", "Hindi", "Spanish", "French",
             "Arabic", "Bengali", "Portuguese", "Russian", "Japanese", "Mandarin Chinese"],
            key="invoice_language",
        )
    with col2:
        with st.expander("💱 Currency Conversion (live rates)", expanded=False):
            src_curr = st.selectbox("Source", ["INR", "USD", "EUR", "JPY", "GBP"], key="src_curr")
            tgt_curr = st.selectbox("Target", ["EUR", "USD", "JPY", "INR", "GBP"], key="tgt_curr")
            apply_conv = st.checkbox("Apply live conversion", key="apply_conv")

    if st.button("⚡ Generate Invoice", use_container_width=True):
        if selected_lang.startswith("Select"):
            st.warning("Please select a language first.")
        else:
            with st.spinner(f"Generating invoice in {selected_lang}…"):
                data = copy.deepcopy(st.session_state.excel_text)
                rate = get_exchange_rate(src_curr, tgt_curr) if apply_conv else 1.0

                if apply_conv and rate != 1.0:
                    for row in data:
                        if "summary" in row:
                            for k in ("subtotal", "tax", "discount", "grand_total"):
                                if k in row["summary"] and row["summary"][k] is not None:
                                    row["summary"][k] = round(row["summary"][k] * rate, 2)
                        else:
                            for field in ("unit_price", "total", "tax", "discount"):
                                if field in row and row[field] is not None:
                                    row[field] = round(row[field] * rate, 2)

                answer = generate_invoice_from_excel(
                    data,
                    language=selected_lang,
                    src_currency=src_curr if apply_conv else None,
                    tgt_currency=tgt_curr if apply_conv else None,
                    rate=rate if apply_conv else None,
                )
            st.session_state.chat.append(("user", f"Generate invoice from {st.session_state.excel_filename} in {selected_lang}"))
            st.session_state.chat.append(("assistant", answer))
            st.rerun()


# ════════════════════════════════════════════════════════════════════
# CHAT INPUT (RAG pipeline)
# ════════════════════════════════════════════════════════════════════
st.markdown("---")
query = st.chat_input("Ask anything about your documents…")

if query:
    st.session_state.chat.append(("user", query))

    if route(query) == "invoice":
        answer = generate_invoice(query)

    elif is_count_query(query):
        answer = count_bills(query)

    else:
        q_lower = query.lower()
        global_keywords = [
            "total amount of", "total amount in", "all", "every",
            "sum of", "how much in total", "entire", "aggregate",
            "how many bills", "count of", "total bills",
            "list all", "all invoices", "combien de factures",
            "cuantos", "quantos",
        ]
        is_global = any(k in q_lower for k in global_keywords)
        k = 12 if is_global else 4
        retrieved_chunks = retrieve(query, k=k)

        if is_global and bill_registry:
            present_bills = set()
            for chunk in retrieved_chunks:
                for bill_name in bill_registry:
                    if f"[{bill_name}]" in chunk:
                        present_bills.add(bill_name)
            for bill_name in bill_registry:
                if bill_name not in present_bills:
                    for doc in documents:
                        if doc.startswith(f"[{bill_name}]"):
                            retrieved_chunks.append(doc)
                            break

        context = "\n".join(retrieved_chunks)

        if is_global and bill_registry:
            bill_list_str = "\n".join(f"- {name}" for name in bill_registry)
            context += f"\n\n[ALL UPLOADED BILL FILES:]\n{bill_list_str}"

        if not context.strip():
            answer = "📭 Please upload a document first so I can answer questions about it."
        else:
            user_lang = get_language(query)
            if is_global:
                prompt = f"""You are a precise assistant. Answer ONLY using the provided context.
Do NOT use any outside knowledge.

The context contains chunks from {len(bill_registry)} different bill documents.
A complete list of all uploaded bill files is at the end (under [ALL UPLOADED BILL FILES:]).

Rules:
- First, check if the question asks about a specific brand, company, or file.
  If so, use ONLY the bills that match.
- If the question does NOT specify a brand, combine ALL bills.
- When calculating totals: show individual amounts then the exact sum.
- When counting bills, count only distinct relevant bills.
- Mention source document(s) AFTER the answer.
- Reply in {user_lang}.

Context:
{context}

Question: {query}
Answer (in {user_lang}):"""
            else:
                prompt = f"""You are a precise assistant. Answer ONLY using the provided document context.
Do NOT use any outside knowledge.

Rules:
- Answer the question directly with exact values.
- Extract amounts, GST, customer names, etc. precisely.
- Mention the document filename AFTER the answer, in parentheses.

Context:
{context}

Question: {query}
Answer (in {user_lang}):"""
            answer = ask_llm(prompt)

    st.session_state.chat.append(("assistant", answer))
    st.rerun()


# ════════════════════════════════════════════════════════════════════
# DISPLAY CHAT
# ════════════════════════════════════════════════════════════════════
if st.session_state.chat:
    for role, msg in st.session_state.chat:
        if role == "user":
            st.markdown(f'<div class="chat-user">🧑 {msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot">🤖 {msg}</div>', unsafe_allow_html=True)
else:
    st.markdown(
        """
        <div style="text-align:center; padding:3rem; opacity:.6;">
            <p style="font-size:3rem;">🧾</p>
            <p style="font-size:1.1rem; font-weight:500;">Upload a document or ask a question to get started</p>
            <p style="font-size:.85rem;">Supports PDF, TXT, Excel, CSV • 10+ languages • Live currency conversion</p>
        </div>
        """,
        unsafe_allow_html=True,
    )