
import os
import streamlit as st
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

st.set_page_config(
    page_title="Ruby on Rails RAG Assistant",
    page_icon="📚",
    layout="centered"
)

MODEL_NAME = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "rails_documentation"
DB_DIR = os.path.join(os.path.dirname(__file__), "rails_chroma_db")

SYSTEM_PROMPT = """
You are a technical documentation assistant specializing in Ruby on Rails.

Answer the user's question using ONLY the provided documentation context.

Rules:
1. Use only information explicitly supported by the context.
2. Do not invent facts, methods, commands, or code examples.
3. If the context does not contain enough information, say:
   "I could not find this information in the provided documentation."
4. Give clear and concise technical explanations.
5. Preserve Ruby on Rails terminology and code syntax.
6. Respond in the same language as the user's question.
7. Do not claim that you searched documents beyond the provided context.

Documentation context:
{context}
"""

@st.cache_resource
def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DB_DIR
    )

def ask_question(question, vector_store, client):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    docs = retriever.invoke(question)

    context = "\n\n".join(
        f"[Source {i}]\n"
        f"Title: {doc.metadata['title']}\n"
        f"URL: {doc.metadata['source']}\n"
        f"Content:\n{doc.page_content}"
        for i, doc in enumerate(docs, 1)
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(context=context)
            },
            {
                "role": "user",
                "content": question
            }
        ],
        temperature=0.2,
        max_tokens=800
    )

    sources = []
    seen = set()

    for doc in docs:
        url = doc.metadata["source"]
        if url not in seen:
            sources.append((doc.metadata["title"], url))
            seen.add(url)

    return response.choices[0].message.content, sources

st.title("📚 Ruby on Rails RAG Assistant")
st.write(
    "Ask questions about Ruby on Rails using information "
    "from the official Rails Guides documentation."
)

try:
    api_key = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY is not configured.")
    st.stop()

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

vector_store = load_vector_store()

question = st.text_input(
    "Your question:",
    placeholder="How do Rails controllers work?"
)

if st.button("Ask", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching documentation and generating answer..."):
            try:
                answer, sources = ask_question(
                    question,
                    vector_store,
                    client
                )

                st.subheader("Answer")
                st.markdown(answer)

                st.subheader("Sources")
                for title, url in sources:
                    st.markdown(f"- [{title}]({url})")

            except Exception as e:
                st.error(f"An error occurred: {e}")
