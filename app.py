import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv
import os
import tempfile

load_dotenv()

st.set_page_config(page_title="PDF Chat", page_icon="📄", layout="centered")

st.markdown("""
    <style>
    .answer-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-top: 10px;
    }
    .question-box {
        background-color: #e8f4fd;
        border-radius: 10px;
        padding: 10px;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📄 Chat with your PDF")
st.caption("Upload a PDF and ask questions about it in plain English.")

@st.cache_resource
def build_chain(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY")
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
    )
    return chain

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain_ready" not in st.session_state:
    st.session_state.chain_ready = False

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    if not st.session_state.chain_ready:
        with st.spinner("Reading and indexing your PDF — this takes a minute on first load..."):
            try:
                chain = build_chain(tmp_path)
                st.session_state.chain = chain
                st.session_state.chain_ready = True
                st.session_state.messages = []
            except Exception as e:
                st.error(f"Failed to process PDF: {str(e)}")

    if st.session_state.chain_ready:
        st.success("✅ PDF ready. Ask anything below.")

        # Display chat history
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="question-box">🧑 {msg["content"]}</div>',
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="answer-box">🤖 {msg["content"]}</div>',
                           unsafe_allow_html=True)

        question = st.chat_input("Ask a question about your PDF...")

        if question:
            st.session_state.messages.append({"role": "user", "content": question})

            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.chain.invoke({"query": question})
                    answer = result["result"]
                except Exception as e:
                    answer = f"Error getting answer: {str(e)}"

            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()
else:
    st.session_state.chain_ready = False
    st.session_state.messages = []
    st.info("👆 Upload a PDF to get started.")