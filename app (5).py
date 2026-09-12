import os
import io
import numpy as np
import streamlit as st
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

# -----------------------------
# Configuration
# -----------------------------
st.set_page_config(
    page_title="PDF Chat with FAISS",
    page_icon="📚",
    layout="wide",
)

GROQ_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Keep chunks small enough for the embedding model.
CHUNK_TOKENS = 200
CHUNK_OVERLAP = 40
TOP_K = 5


# -----------------------------
# Cached models
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def load_groq_client():
    api_key = None

    # Streamlit Cloud / local .streamlit/secrets.toml
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    # Optional local environment variable
    api_key = api_key or os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


# -----------------------------
# PDF extraction
# -----------------------------
def extract_pdf_text(pdf_bytes: bytes):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())

        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

    return pages


# -----------------------------
# Token-based chunking
# -----------------------------
def chunk_pages(pages, tokenizer):
    chunks = []

    for page in pages:
        text = page["text"]

        # Tokenize without truncating.
        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )

        start = 0

        while start < len(token_ids):
            end = min(start + CHUNK_TOKENS, len(token_ids))
            chunk_ids = token_ids[start:end]

            chunk_text = tokenizer.decode(
                chunk_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "page": page["page"],
                        "chunk_id": len(chunks),
                    }
                )

            if end >= len(token_ids):
                break

            start = end - CHUNK_OVERLAP

    return chunks


# -----------------------------
# FAISS index
# -----------------------------
def create_faiss_index(chunks, embedding_model):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    # With normalized vectors, inner product == cosine similarity.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings


def search_faiss(query, index, chunks, embedding_model, top_k=TOP_K):
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    k = min(top_k, index.ntotal)

    scores, indices = index.search(query_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        result = dict(chunks[idx])
        result["score"] = float(score)
        results.append(result)

    return results


# -----------------------------
# Groq answer generation
# -----------------------------
def generate_answer(question, search_results, groq_client):
    context_parts = []

    for result in search_results:
        context_parts.append(
            f"[Page {result['page']}]\n{result['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """You are a PDF question-answering assistant.

Answer the user's question using ONLY the provided PDF context.
If the answer is not present in the context, say:
"I couldn't find that information in the uploaded PDF."

Do not invent facts.
When possible, mention the page number(s) supporting the answer.
Keep the answer clear and reasonably concise.
"""

    user_prompt = f"""PDF CONTEXT:

{context}

QUESTION:
{question}
"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1000,
    )

    return response.choices[0].message.content


# -----------------------------
# UI
# -----------------------------
st.title("📚 PDF Chat — FAISS + Open-Source Embeddings + Groq")
st.caption(
    "Upload a PDF → extract text → tokenize → chunk → embed → index with FAISS → ask questions."
)

with st.sidebar:
    st.header("Settings")
    st.write(f"**LLM:** `{GROQ_MODEL}`")
    st.write(f"**Embedding model:** `{EMBEDDING_MODEL}`")
    st.write(f"**Chunk size:** `{CHUNK_TOKENS}` tokens")
    st.write(f"**Chunk overlap:** `{CHUNK_OVERLAP}` tokens")
    st.write(f"**Retrieved chunks:** `{TOP_K}`")

groq_client = load_groq_client()

if groq_client is None:
    st.warning(
        "GROQ_API_KEY is not configured. Add it to Streamlit Secrets before asking questions."
    )

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"],
    help="For the first version, use text-based PDFs. Scanned/image-only PDFs need OCR.",
)

if uploaded_file:
    if st.session_state.get("file_name") != uploaded_file.name:
        # New PDF: rebuild the in-memory index.
        st.session_state.pop("index", None)
        st.session_state.pop("chunks", None)
        st.session_state.pop("file_name", None)

    if "index" not in st.session_state:
        with st.spinner("Loading embedding model..."):
            embedding_model = load_embedding_model()

        with st.spinner("Extracting PDF text..."):
            pdf_bytes = uploaded_file.getvalue()
            pages = extract_pdf_text(pdf_bytes)

        if not pages:
            st.error(
                "No selectable text was found. This may be a scanned PDF. "
                "OCR support can be added as a next step."
            )
            st.stop()

        with st.spinner("Tokenizing and creating chunks..."):
            tokenizer = embedding_model.tokenizer
            chunks = chunk_pages(pages, tokenizer)

        if not chunks:
            st.error("No usable text chunks were created.")
            st.stop()

        with st.spinner("Creating embeddings and FAISS index..."):
            index, _ = create_faiss_index(chunks, embedding_model)

        st.session_state["index"] = index
        st.session_state["chunks"] = chunks
        st.session_state["file_name"] = uploaded_file.name

        st.success(
            f"Processed **{uploaded_file.name}**: "
            f"{len(pages)} pages → {len(chunks)} chunks → FAISS index created."
        )

    else:
        st.success(
            f"Ready: **{st.session_state['file_name']}** "
            f"({len(st.session_state['chunks'])} chunks indexed)"
        )

    question = st.text_input(
        "Ask a question about the PDF",
        placeholder="e.g. What are the main conclusions?",
    )

    if st.button("Ask", type="primary", disabled=not question.strip()):
        if groq_client is None:
            st.error("Configure GROQ_API_KEY first.")
            st.stop()

        with st.spinner("Searching the PDF..."):
            embedding_model = load_embedding_model()

            results = search_faiss(
                question,
                st.session_state["index"],
                st.session_state["chunks"],
                embedding_model,
                TOP_K,
            )

        with st.spinner("Generating answer..."):
            answer = generate_answer(
                question,
                results,
                groq_client,
            )

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Retrieved PDF chunks"):
            for i, result in enumerate(results, start=1):
                st.markdown(
                    f"**{i}. Page {result['page']} — similarity {result['score']:.3f}**"
                )
                st.write(result["text"])

else:
    st.info("Upload a PDF to begin.")
