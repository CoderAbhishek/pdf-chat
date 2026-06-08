from pypdf import PdfReader

reader = PdfReader("sample.pdf")
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if "revenue" in text.lower() or "94" in text:
        print(f"--- Page {i+1} ---")
        print(text[:1000])
        print()