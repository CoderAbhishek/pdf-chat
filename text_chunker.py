from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    chunks = splitter.split_text(text)
    return chunks

if __name__ == "__main__":
    text = load_pdf("sample.pdf")
    chunks = split_text(text)
    
    print(f"Total characters in document: {len(text)}")
    print(f"Total chunks created: {len(chunks)}")
    print(f"\n--- Chunk 1 ---")
    print(chunks[0])
    print(f"\n--- Chunk 2 ---")
    print(chunks[1])
    print(f"\n--- Last chunk ---")
    print(chunks[-1])