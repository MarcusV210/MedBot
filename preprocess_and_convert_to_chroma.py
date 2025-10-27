from langchain_community.document_loaders import PyPDFLoader #type: ignore
from langchain_community.vectorstores import Chroma #type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter #type: ignore
from langchain_community.embeddings import HuggingFaceEmbeddings #type: ignore
import re, os

textbook_path = "data/Harrisons Principles of Internal Medicine 21st Edition.pdf"
loader = PyPDFLoader(textbook_path)
docs = loader.load()

docs = docs[168:-1201] # Removing the index and last citation pages. There are 13,796 useful pages.
# docs = docs[168:2000]
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

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", "!", "?"]
)

split_docs = splitter.split_documents(cleaned_docs)
print(f"4. Split into {len(split_docs)} smaller chunks.")


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

persist_dir = "Harrison_DB"

if not os.path.exists(persist_dir):
    os.makedirs(persist_dir)

db = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding_model,
    persist_directory=persist_dir
)

db.persist()

print(f"5. ChromaDB successfully created at: {persist_dir}")
print("All textbook pages stored and ready for retrieval.")