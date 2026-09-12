```python
import os
import re
import io
import requests
import numpy as np
import streamlit as st
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# ============================================================
# CyerLawGPT
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

DEFAULT_MODEL = "grok-4.6"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 180
CHUNK_OVERLAP = 35


# ============================================================
# PAGE TITLE
# ============================================================

st.title("⚖️ CyerLawGPT")

st.markdown(
    """
    **Pakistan Cyber Law AI Assistant**

    Ask questions about Pakistan's cybercrime law using the supplied
    legal document as the primary source.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    api_key = st.text_input(
        "Grok / xAI API Key",
        value=os.getenv("XAI_API_KEY", ""),
        type="password",
        help="Enter your xAI API key."
    )

    model_name = st.text_input(
        "Grok Model",
        value=os.getenv("XAI_MODEL", DEFAULT_MODEL)
    )

    technical_level = st.selectbox(
        "Technical / Legal Level",
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

    st.caption(
        "⚠️ This tool provides educational information and is "
        "not a substitute for advice from a qualified lawyer."
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_google_doc_id(url):
    """
    Extract Google Docs document ID.
    """

    match = re.search(
        r"/document/d/([a-zA-Z0-9_-]+)",
        url
    )

    if match:
        return match.group(1)

    return None


def get_pdf_url():

    custom_pdf = os.getenv("CYBER_LAW_PDF_URL", "").strip()

    if custom_pdf:
        return custom_pdf

    doc_url = os.getenv(
        "CYBER_LAW_DOC_URL",
        DEFAULT_DOC_URL
    )

    doc_id = extract_google_doc_id(doc_url)

    if doc_id:
        return (
            f"https://docs.google.com/document/d/"
            f"{doc_id}/export?format=pdf"
        )

    return doc_url


@st.cache_data(show_spinner=False)
def download_pdf():

    pdf_url = get_pdf_url()

    response = requests.get(
        pdf_url,
        timeout=60
    )

    response.raise_for_status()

    content_type = response.headers.get(
        "content-type",
        ""
    ).lower()

    if (
        "pdf" not in content_type
        and not response.content.startswith(b"%PDF")
    ):
        raise ValueError(
            "The supplied document URL did not return a PDF. "
            "Check CYBER_LAW_DOC_URL or CYBER_LAW_PDF_URL."
        )

    return response.content


@st.cache_data(show_spinner=False)
def extract_pdf_text(pdf_bytes):

    reader = PdfReader(
        io.BytesIO(pdf_bytes)
    )

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

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


@st.cache_resource(show_spinner=False)
def load_embedding_model():

    model_name = os.getenv(
        "EMBEDDING_MODEL",
        DEFAULT_EMBEDDING_MODEL
    )

    return SentenceTransformer(
        model_name
    )


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


def build_system_prompt():

    if grounding == "Maximum strictness":

        grounding_instruction = """
        Use ONLY information supported by the retrieved legal passages.

        Do not infer missing legal provisions.

        If the retrieved material does not answer the question,
        explicitly say that the supplied document does not contain
        enough information to answer it reliably.
        """

    elif grounding == "Strictly source-grounded":

        grounding_instruction = """
        Base legal claims primarily on the retrieved passages.

        Do not invent section numbers, punishments, definitions,
        procedures, authorities, or legal requirements.
        """

    else:

        grounding_instruction = """
        Use the retrieved legal passages as the primary source.
        General explanation may be used only when it does not
        contradict the supplied legal material.
        """

    return f"""
You are CyerLawGPT, an AI assistant specializing in Pakistan
cyber law and the Prevention of Electronic Crimes Act (PECA).

Your job is to explain the supplied legal document clearly.

IMPORTANT RULES:

1. Treat the supplied legal document as the primary legal source.

2. {grounding_instruction}

3. Never fabricate:
   - section numbers
   - legal provisions
   - punishments
   - fines
   - imprisonment terms
   - authorities
   - procedures
   - definitions

4. When possible, mention the relevant section number.

5. Clearly distinguish between:
   - what the Act/document says
   - your plain-language explanation

6. If the document does not provide enough information,
   say so instead of guessing.

7. Do not claim to be a lawyer.

8. Do not provide instructions for committing,
   concealing, evading detection of, or optimizing cybercrime.

9. Answer at the requested level:
   {technical_level}

10. Response length:
   {response_size}

11. Response language:
   {language}

Keep explanations accurate, clear, and useful.
"""


def ask_grok(
    question,
    context
):

    if not api_key:

        raise ValueError(
            "Please enter your Grok / xAI API key in the sidebar."
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1"
    )

    context_text = "\n\n".join(
        [
            (
                f"[PDF Page {item['page']}]\n"
                f"{item['text']}"
            )
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
LEGAL DOCUMENT PASSAGES:

{context_text}

--------------------------------------------------

USER QUESTION:

{question}

--------------------------------------------------

Answer the user's question using the legal passages above.
"""
        }
    ]

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.2
    )

    return response.choices[0].message.content


# ============================================================
# LOAD LEGAL DOCUMENT
# ============================================================

try:

    with st.spinner(
        "📚 Downloading legal document and building FAISS index..."
    ):

        index, chunks = build_faiss_index()

        embedding_model = load_embedding_model()

    st.success(
        f"✅ Legal knowledge base ready — {len(chunks)} passages indexed."
    )

except Exception as e:

    st.error(
        "❌ Could not load the legal document."
    )

    st.code(
        str(e)
    )

    st.info(
        "Check your Google Docs URL or set CYBER_LAW_PDF_URL "
        "to a direct PDF URL."
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
# USER QUESTION
# ============================================================

question = st.chat_input(
    "Ask a question about Pakistan cyber law..."
)


if question:

    # Display user question
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

    # Generate answer
    with st.chat_message("assistant"):

        try:

            with st.spinner(
                "🤖 Asking Grok..."
            ):

                answer = ask_grok(
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

            answer = (
                "I could not generate an answer. "
                "Please check your Grok/xAI API key and model."
            )

            st.error(
                answer
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
    "CyerLawGPT • RAG + FAISS + Sentence Transformers + Grok/xAI"
)

st.caption(
    "⚠️ For educational/informational purposes only. "
    "This application does not constitute legal advice."
)
```
