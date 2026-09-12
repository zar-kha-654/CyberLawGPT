# ⚖️ CyerLawGPT

CyerLawGPT is an AI-powered Pakistan Cyber Law assistant built with:

- Python
- Streamlit
- FAISS
- Sentence Transformers
- Grok / xAI API
- PyPDF

The application uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant passages from the supplied Pakistan cyber-law PDF before asking Grok to generate an answer.

---

## 🚀 Features

### 1. Pakistan Cyber Law RAG

CyerLawGPT retrieves relevant passages from the supplied Pakistan cyber-law document and uses those passages as the legal context for Grok.

This helps reduce hallucination and keeps answers grounded in the provided law.

---

### 2. Automatic PDF Download

The application automatically downloads the cyber-law document when it starts.

Default document:

https://docs.google.com/document/d/1YwgWc8Olvd5d5PXbnJUcNhalSItxQsUd68imK5EdY-g/edit?usp=sharing

The Google document must be publicly accessible.

The application converts the Google document into a PDF automatically.

---

### 3. Automatic Embeddings

When the application starts:

1. The PDF is downloaded.
2. Text is extracted from the PDF.
3. The text is divided into smaller passages.
4. Sentence Transformer embeddings are generated locally.
5. FAISS creates a vector index.
6. The index is used for semantic search.

No paid embedding API is required.

---

## 🧠 RAG Architecture

```text
Pakistan Cyber Law PDF
        ↓
     PDF Text
        ↓
     Chunking
        ↓
Sentence Transformer
        ↓
   Embeddings
        ↓
      FAISS
        ↓
User Question
        ↓
Question Embedding
        ↓
Relevant Legal Passages
        ↓
       Grok
        ↓
Grounded Legal Answer
