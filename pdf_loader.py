from pypdf import PdfReader

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Test it
if __name__ == "__main__":
    text = load_pdf("sample.pdf")
    print(f"Total characters extracted: {len(text)}")
    print("\nFirst 500 characters:")
    print(text[:500])