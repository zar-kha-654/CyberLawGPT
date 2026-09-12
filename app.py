import os
import re
import io
import requests
import numpy as np
import streamlit as st
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# CYERLAWGPT
# Pakistan Cyber Law RAG Application
# ============================================================

st.set_page_config(
    page_title="CyerLawGPT",
    page_icon="⚖️",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_DOC_URL = (
    "https://docs.google.com/document/d/"
    "1YwgWc8Olvd5d5PXbnJUcNhalSItxQsUd68imK5EdY-g"
    "/edit?usp=sharing"
)

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 180
CHUNK_OVERLAP = 35


# ============================================================
# TITLE
# ============================================================

st.title("⚖️ CyerLawGPT")

st.markdown(
    """
    ### Pakistan Cyber Law AI Assistant

    Ask questions about Pakistan's cyber law using the supplied
    legal document as the primary source.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    api_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Enter your Groq API key."
    )

    model_name = st.selectbox(
        "AI Model",
        [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant"
        ]
    )

    technical_level = st.selectbox(
        "Answer Level",
        [
            "Beginner",
            "Intermediate",
            "Technical",
            "Legal-professional"
        ]
    )

    response_size = st.selectbox(
        "Response Size",
        [
            "Short",
            "Medium",
            "Detailed",
            "Very detailed"
        ]
    )

    language = st.selectbox(
        "Response Language",
        [
            "English",
            "Urdu",
            "Roman Urdu"
        ]
    )

    grounding = st.selectbox(
        "Legal Grounding",
        [
            "Balanced",
            "Strictly source-grounded",
            "Maximum strictness"
        ]
    )

    top_k = st.slider(
        "Retrieved Passages",
        min_value=3,
        max_value=10,
        value=5
    )

    show_sources = st.checkbox(
        "Show retrieved legal passages",
        value=False
    )

    st.divider()

    st.warning(
        "This application provides educational information "
        "and is not a substitute for advice from a qualified lawyer."
    )


# ============================================================
# GOOGLE DOC ID
# ============================================================

def extract_google_doc_id(url):

    match = re.search(
        r"/document/d/([a-zA-Z0-9_-]+)",
        url
    )

    if match:
        return match.group(1)

    return None


# ============================================================
# GET PDF URL
# ============================================================

def get_pdf_url():

    custom_pdf = os.getenv(
        "CYBER_LAW_PDF_URL",
        ""
    ).strip()

    if custom_pdf:
        return custom_pdf

    doc_url = os.getenv(
        "CYBER_LAW_DOC_URL",
        DEFAULT_DOC_URL
    )

    doc_id = extract_google_doc_id(
        doc_url
    )

    if doc_id:

        return (
            "https://docs.google.com/document/d/"
            + doc_id
            + "/export?format=pdf"
        )

    return doc_url


# ============================================================
# DOWNLOAD PDF
# ============================================================

@st.cache_data(show_spinner=False)
def download_pdf():

    pdf_url = get_pdf_url()

    response = requests.get(
        pdf_url,
        timeout=60
    )

    response.raise_for_status()

    if not response.content.startswith(b"%PDF"):

        raise ValueError(
            "The supplied document URL did not return a valid PDF. "
            "Make sure the Google Doc is publicly accessible."
        )

    return response.content


# ============================================================
# EXTRACT PDF TEXT
# ============================================================

@st.cache_data(show_spinner=False)
def extract_pdf_text(pdf_bytes):

    reader = PdfReader(
        io.BytesIO(pdf_bytes)
    )

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text() or ""

        text = text.replace(
            "\x00",
            " "
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        if text:

            pages.append(
                {
                    "page": page_number,
                    "text": text
                }
            )

    return pages


# ============================================================
# CREATE CHUNKS
# ============================================================

def create_chunks(pages):

    chunks = []

    for page in pages:

        words = page["text"].split()

        if not words:
            continue

        start = 0

        while start < len(words):

            end = min(
                start + CHUNK_SIZE,
                len(words)
            )

            chunk_text = " ".join(
                words[start:end]
            )

            chunks.append(
                {
                    "page": page["page"],
                    "text": chunk_text
                }
            )

            if end >= len(words):
                break

            start = max(
                end - CHUNK_OVERLAP,
                start + 1
            )

    return chunks


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model():

    model_name = os.getenv(
        "EMBEDDING_MODEL",
        DEFAULT_EMBEDDING_MODEL
    )

    return SentenceTransformer(
        model_name
    )


# ============================================================
# BUILD FAISS INDEX
# ============================================================

@st.cache_resource(show_spinner=False)
def build_faiss_index():

    pdf_bytes = download_pdf()

    pages = extract_pdf_text(
        pdf_bytes
    )

    chunks = create_chunks(
        pages
    )

    if not chunks:

        raise ValueError(
            "No readable text was found in the PDF."
        )

    model = load_embedding_model()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(
        embeddings
    )

    return index, chunks


# ============================================================
# RETRIEVE RELEVANT PASSAGES
# ============================================================

def retrieve_context(
    question,
    index,
    chunks,
    model,
    top_k
):

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype="float32"
    )

    scores, indices = index.search(
        question_embedding,
        min(top_k, len(chunks))
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        results.append(
            {
                "page": chunks[idx]["page"],
                "text": chunks[idx]["text"],
                "score": float(score)
            }
        )

    return results


# ============================================================
# SYSTEM PROMPT
# ============================================================

def build_system_prompt():

    if grounding == "Maximum strictness":

        grounding_instruction = """
Use ONLY information supported by the retrieved legal passages.

If the retrieved passages do not contain enough information,
clearly say that the supplied document does not contain
enough information to answer the question reliably.

Never guess.
"""

    elif grounding == "Strictly source-grounded":

        grounding_instruction = """
Base legal claims primarily on the retrieved legal passages.

Do not invent section numbers, punishments, fines,
definitions, authorities, or procedures.
"""

    else:

        grounding_instruction = """
Use the retrieved legal passages as the primary source.
You may explain the material in simple language, but
do not contradict the supplied legal document.
"""

    return f"""
You are CyerLawGPT, an AI assistant specializing in
Pakistan cyber law.

Your primary task is to explain the supplied legal document,
including the Prevention of Electronic Crimes Act (PECA),
in an accurate and understandable way.

IMPORTANT RULES:

{grounding_instruction}

Never fabricate:

- Section numbers
- Punishments
- Fines
- Imprisonment terms
- Definitions
- Legal procedures
- Authorities
- Legal requirements

When possible, mention the relevant section number.

Clearly distinguish between:

1. What the law/document says.
2. Your plain-language explanation.

If the document does not contain enough information,
say so instead of guessing.

Do not claim to be a lawyer.

Do not provide instructions for committing,
concealing, evading detection of, or optimizing cybercrime.

Answer according to this level:

{technical_level}

Use this response length:

{response_size}

Use this language:

{language}

Be accurate, clear, and helpful.
"""


# ============================================================
# ASK GROQ
# ============================================================

def ask_groq(
    question,
    context
):

    if not api_key:

        raise ValueError(
            "Please enter your Groq API key in the sidebar."
        )

    client = Groq(
        api_key=api_key
    )

    context_text = "\n\n".join(
        [
            f"[PDF Page {item['page']}]\n{item['text']}"
            for item in context
        ]
    )

    messages = [
        {
            "role": "system",
            "content": build_system_prompt()
        },
        {
            "role": "user",
            "content": f"""
Here are the relevant passages retrieved from
the Pakistan cyber law document:

==================================================

{context_text}

==================================================

USER QUESTION:

{question}

==================================================

Answer the user's question using the retrieved
legal passages as your primary source.
"""
        }
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.2,
        max_tokens=2000
    )

    return response.choices[0].message.content


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

try:

    with st.spinner(
        "📚 Downloading legal PDF and building FAISS knowledge base..."
    ):

        index, chunks = build_faiss_index()

        embedding_model = load_embedding_model()

    st.success(
        f"✅ Knowledge base ready — {len(chunks)} passages indexed."
    )

except Exception as e:

    st.error(
        "❌ Could not load the legal document."
    )

    st.code(
        str(e)
    )

    st.info(
        "Make sure the Google Doc is publicly accessible "
        "and contains the legal document."
    )

    st.stop()


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about Pakistan cyber law..."
)


if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(
            question
        )

    # Retrieve relevant passages
    with st.spinner(
        "🔎 Searching the legal document..."
    ):

        retrieved = retrieve_context(
            question,
            index,
            chunks,
            embedding_model,
            top_k
        )

    # Ask Groq
    with st.chat_message("assistant"):

        try:

            with st.spinner(
                "🤖 Groq is generating the answer..."
            ):

                answer = ask_groq(
                    question,
                    retrieved
                )

            st.markdown(
                answer
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

        except Exception as e:

            st.error(
                "❌ Groq API request failed."
            )

            st.code(
                str(e)
            )


# ============================================================
# SHOW SOURCES
# ============================================================

if question and show_sources:

    st.divider()

    st.subheader(
        "📄 Retrieved Legal Passages"
    )

    for i, item in enumerate(
        retrieved,
        start=1
    ):

        with st.expander(
            f"Passage {i} — PDF Page {item['page']} "
            f"— Similarity {item['score']:.3f}"
        ):

            st.write(
                item["text"]
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CyerLawGPT • RAG + FAISS + Sentence Transformers + Groq"
)

st.caption(
    "⚠️ For educational/informational purposes only. "
    "This application does not constitute legal advice."
)
