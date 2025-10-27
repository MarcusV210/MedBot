import pandas as pd #type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline #type: ignore
from huggingface_hub import login #type: ignore
from dotenv import load_dotenv #type: ignore
from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
from langchain_community.vectorstores import Chroma # type: ignore
import os

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
login(HF_TOKEN)
print("1. Logged in.")

FAQ = pd.read_csv('data/FAQ_Test.csv')

PERSIST_DIR = "Harrison_DB" 
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model_name = "google/gemma-3-1b-it"
K = 5  

print("2. Loading embedding function")
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)


print("3. Model loading...........")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300,
    temperature=0.7,
    do_sample=False
)

questions = FAQ['Question'].tolist()

questions_with_context = []
for q in questions:
    query_vec = embedding_model.embed_query(q)
    context_docs = db.similarity_search_by_vector(query_vec, k=K)
    context = "\n".join([d.page_content for d in context_docs])

    prompt = f"""You are a helpful medical bot who helps students learn. Write a detailed, paragraph-style explanation for the following question with the given context. Avoid bullet points or lists.
    
    Question: {q}
    Context: {context}
    Answer:
    """

    questions_with_context.append(prompt)

print("4. All questions embeded.")


responses = pipe(
    questions,          
    max_new_tokens=200,
    temperature=0.7,
    top_p = 0.9
)

test_answers = [r[0]["generated_text"] for r in responses]
FAQ["Answer"] = test_answers
print(f"5. RAG done and FAQ dataset updated.")
FAQ.to_csv('data/rag/FAQ_responses.csv', index=False)

def clean_answers(df, question_col="Question", answer_col="Answer"):
    cleaned_answers = []

    for q, a in zip(df[question_col], df[answer_col]):
        if not isinstance(a, str):
            cleaned_answers.append(a)
            continue

        # Remove exact question text if present at start
        cleaned = a
        if cleaned.startswith(q):
            cleaned = cleaned[len(q):]

        # Strip extra whitespace and punctuation at the start
        cleaned = cleaned.lstrip(" :.-\n").strip()
        cleaned_answers.append(cleaned)

    df[answer_col] = cleaned_answers
    return df

FAQ_Cleaned = clean_answers(FAQ)
FAQ_Cleaned.to_csv("data/rag/FAQ_cleaned_responses.csv", index=False)