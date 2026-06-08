from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)

def create_vector_store(chunks):
    # Load the embedding model locally — downloads once, ~90MB
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Create embeddings for all chunks
    embeddings = model.encode(chunks)
    
    # Set up ChromaDB
    client = chromadb.Client()
    collection = client.create_collection("pdf_collection")
    
    # Store chunks + embeddings in ChromaDB
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    
    return collection, model

def query_vector_store(collection, model, query, n_results=3):
    query_embedding = model.encode([query])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results
    )
    return results["documents"][0]

if __name__ == "__main__":
    print("Loading PDF...")
    text = load_pdf("sample.pdf")
    
    print("Splitting into chunks...")
    chunks = split_text(text)
    print(f"Created {len(chunks)} chunks")
    
    print("Creating embeddings and storing in ChromaDB...")
    collection, model = create_vector_store(chunks)
    print("Vector store ready")
    
    # Test retrieval — no LLM yet, just finding relevant chunks
    query = "What is generative AI?"
    print(f"\nQuery: {query}")
    print("\nTop 3 relevant chunks retrieved:")
    relevant_chunks = query_vector_store(collection, model, query)
    for i, chunk in enumerate(relevant_chunks):
        print(f"\n--- Result {i+1} ---")
        print(chunk[:300])