from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()

def build_rag_chain(pdf_path):
    # Step 1: Load
    loader = PDFPlumberLoader(pdf_path)
    documents = loader.load()

    # Step 2: Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks: {len(chunks)}")

    # Step 3: Embed + Store
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(chunks, embeddings)

    # Step 4: Build retrieval chain
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY")
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
    )
    return chain

if __name__ == "__main__":
    print("Building RAG chain...")
    chain = build_rag_chain("sample.pdf")
    print("Ready.\n")

    questions = [
        "What was Tesla's total revenue?",
        "What are the main risks mentioned in the report?",
        "How many vehicles did Tesla deliver?"
    ]

    for question in questions:
        result = chain.invoke({"query": question})
        print(f"Question: {question}")
        print(f"Answer: {result['result']}")
        print("-" * 50)