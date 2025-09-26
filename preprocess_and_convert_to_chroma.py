from langchain_community.document_loaders import PyPDFLoader #type: ignore
import re

textbook_path = 'data/Harrisons Principles of Internal Medicine 21st Edition.pdf'
loader = PyPDFLoader(textbook_path)
docs = loader.load()

docs = docs[168:-1201] # Removing the index and last citation pages. There are 13,796 useful pages.
print("1.Document loaded.")

cleaned_docs = []
for page in docs:
    lines = page.page_content.split("\n")
    cleaned = []

    for line in lines:
        if not line.strip():  # Skip empty lines
            continue
        if re.match(r'^\s*\d+\s*$', line):  # just page numbers
            continue
        if re.match(r'^\s*Page\s+\d+(\s+of\s+\d+)?\s*$', line, re.IGNORECASE):
            continue
        if re.match(r'^[A-Z\s]{5,}$', line):  # ALL CAPS headers
            continue
        if re.match(r'^[-–—_.]+$', line):  # divider lines
            continue

        cleaned.append(line)

    page.page_content = "\n".join(cleaned)
    cleaned_docs.append(page)

print("2. Document Cleaned.")

def clean_content(text):
    text = re.sub(r'(\w+)-\n+(\w+)', r'\1\2', text) # Remove hiphens
    text = re.sub(r'\n+', ' ', text) # Remove new lines
    text = re.sub(r'\s{2,}', ' ', text) # Remove too many spaces

    return text.strip()

for page in cleaned_docs:
    page.page_content = clean_content(page.page_content)

print("3. Pages cleaned of any weird characters.")