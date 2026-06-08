# from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
from dotenv import load_dotenv
import os
import pdfplumber

load_dotenv()

def load_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)

def create_vector_store(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    client = chromadb.Client()
    collection = client.create_collection("pdf_collection")
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    return collection, model

def retrieve_chunks(collection, model, query, n_results=3):
    query_embedding = model.encode([query])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results
    )
    return results["documents"][0]

def generate_answer(query, relevant_chunks):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    context = "\n\n".join(relevant_chunks)
    
    prompt = f"""Answer the question using only the context below. 
Be concise. Maximum 3 sentences. 
If the answer is not in the context, say only: 
"I don't have enough information to answer that."
Do not add anything else.

Context:
{context}

Question: {query}
Answer:"""
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

def ask_pdf(collection, model, query):
    relevant_chunks = retrieve_chunks(collection, model, query)
    answer = generate_answer(query, relevant_chunks)
    return answer

if __name__ == "__main__":
    print("Setting up RAG pipeline...")
    text = load_pdf("sample.pdf")
    chunks = split_text(text)
    collection, model = create_vector_store(chunks)
    print("Ready. Ask me anything about your PDF.\n")
    
    # Test with 3 questions
    questions = [
    "What was Tesla's total revenue?",
    "What are the main risks mentioned in the report?",
    "How many vehicles did Tesla deliver?"
    ]
    
    for question in questions:
        print(f"Question: {question}")
        answer = ask_pdf(collection, model, question)
        print(f"Answer: {answer}")
        print("-" * 50)