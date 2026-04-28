# 🧾 AI-Powered Multilingual Invoice & Bill Assistant

> An intelligent, end-to-end document assistant that lets you upload financial documents, ask questions about them in any language, and instantly generate professional invoices — with live currency conversion.

---

## 📌 Overview

The **AI Multilingual Invoice & Bill Assistant** is a full-stack AI application that combines **Retrieval-Augmented Generation (RAG)**, **Large Language Models (LLaMA 3 via Groq)**, and **smart document processing** to help users:

- 📤 Upload bills and invoices (PDF, Excel, CSV, TXT)
- 💬 Ask natural-language questions about their documents (in any language)
- 🧾 Automatically generate formatted, professional invoices
- 🌍 Get responses in **10+ languages**
- 💱 Convert currency values using **live exchange rates**

---

## 🚀 Features

| Feature | Details |
|---|---|
| **Multi-format Ingestion** | Supports PDF, Excel (.xlsx/.xls), CSV, and TXT |
| **RAG Pipeline** | FAISS vector store + Sentence Transformers for semantic search |
| **LLM Integration** | Groq API (LLaMA 3.1 8B / 3.3 70B) with retry logic |
| **Invoice Generation** | From natural language queries OR structured Excel data |
| **Multilingual Support** | 10+ languages: English, Hindi, Spanish, French, Arabic, Bengali, Portuguese, Russian, Japanese, Mandarin |
| **Auto Language Detection** | Detects user query language and responds in the same language |
| **Smart Excel Processing** | Auto header-row detection, LLM-assisted column mapping, currency cleaning |
| **Live Currency Conversion** | Real-time rates via Frankfurter API (INR, USD, EUR, GBP, JPY) |
| **State-Persistent Chat** | Streamlit session-state powered chat interface |
| **Premium UI** | Dark theme with glassmorphism, gradient animations, and responsive design |

---

## 🏗️ Architecture

```
User Upload (PDF / Excel / TXT)
        │
        ▼
┌─────────────────────┐
│  Document Processor │  ← pdfplumber / pandas / smart header detection
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   RAG Pipeline      │  ← Sentence Transformers → FAISS Index
│  (rag_pipeline.py)  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐     ┌──────────────────────┐
│   Query Router      │────▶│  Invoice Generator   │
│   (router.py)       │     │ (invoice_generator.py)│
└────────┬────────────┘     └──────────────────────┘
         │
         ▼
┌─────────────────────┐
│   Groq LLM API      │  ← LLaMA 3.1 8B / 3.3 70B
│      (llm.py)       │
└────────┬────────────┘
         │
         ▼
   Streamlit Chat UI  ← Multilingual response in user's language
```

---

## 🗂️ Project Structure

```
📁 Multilingual-Invoice-and-Bill-Analysis-Assistant/
├── app.py                  # Main Streamlit app (UI, file upload, chat)
├── llm.py                  # Groq LLM API wrapper with retry logic
├── rag_pipeline.py         # FAISS vector store & semantic retrieval
├── invoice_generator.py    # Invoice generation (natural language + Excel)
├── router.py               # Query intent classifier (invoice vs. Q&A)
├── requirements.txt        # Python dependencies
├── .gitignore              # Excludes .env, __pycache__, etc.
└── README.md               # This file
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Kaustubh1020/Multilingual-Invoice-and-Bill-Analysis-Assistant.git
cd Multilingual-Invoice-and-Bill-Analysis-Assistant
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure your API key
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```
> Get your free API key at [console.groq.com](https://console.groq.com)

### 5. Run the application
```bash
streamlit run app.py
```

---

## 🧠 How It Works

### 📄 Document Ingestion
- **PDF**: Text extracted using `pdfplumber`, chunked and indexed into FAISS
- **Excel/CSV**: Smart header-row detection using multilingual keyword matching across 12+ languages; columns are normalized via rule-based and LLM-assisted mapping
- **TXT**: Raw text is chunked and added directly to the vector store

### 🔍 RAG-Powered Q&A
1. User query is embedded using `sentence-transformers`
2. Top-k semantically similar chunks are retrieved from the FAISS index
3. Retrieved context + query are sent to the Groq LLM
4. Response is generated in the **user's detected language**

### 🧾 Invoice Generation
- **From chat**: Natural language requests (e.g. *"Generate an invoice for 5 chairs at ₹200 each"*) are routed to the invoice generator
- **From Excel**: Uploaded spreadsheet data is parsed, normalized, and passed to the LLM to produce a formatted markdown invoice in the selected language

### 💱 Currency Conversion
Live exchange rates are fetched from the [Frankfurter API](https://www.frankfurter.app/). Values in the invoice are pre-converted before being sent to the LLM, with strict prompt constraints to prevent the model from modifying amounts.

---

## 🌍 Supported Languages

| Language | Language | Language |
|---|---|---|
| 🇬🇧 English | 🇮🇳 Hindi | 🇪🇸 Spanish |
| 🇫🇷 French | 🇸🇦 Arabic | 🇧🇩 Bengali |
| 🇧🇷 Portuguese | 🇷🇺 Russian | 🇯🇵 Japanese |
| 🇨🇳 Mandarin Chinese | | |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit (custom dark theme CSS) |
| **LLM** | Groq API — LLaMA 3.1 8B Instant / LLaMA 3.3 70B Versatile |
| **Vector Store** | FAISS (`faiss-cpu`) |
| **Embeddings** | Sentence Transformers (`all-MiniLM-L6-v2`) |
| **PDF Parsing** | pdfplumber |
| **Spreadsheet** | pandas, openpyxl |
| **Language Detection** | langdetect |
| **Currency API** | Frankfurter (free, no key required) |
| **Environment** | python-dotenv |

---

## 📸 Screenshots

> *Upload a bill, ask questions, and generate invoices — all from one interface.*

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👨‍💻 Author

**Kaustubh** — [GitHub](https://github.com/Kaustubh1020)

---

> ⭐ If you found this project useful, consider giving it a star!
