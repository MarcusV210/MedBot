import pandas as pd #type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline #type: ignore
from huggingface_hub import login #type: ignore
from dotenv import load_dotenv #type: ignore
from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
from langchain_community.vectorstores import Chroma # type: ignore
import os
import re
import json
import evaluate # type: ignore

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
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME) #Same as the one used to create the ChromaDB
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
    top_p = 0.9,
    do_sample=False
)

questions = FAQ['Question'].tolist()
# questions = questions[:2] #just the first two values to test

def clean_text(text: str) -> str:

    text = text.encode('utf-8', 'ignore').decode('unicode_escape')
    text = re.sub(r'[\r\n\t\f\v]+', ' ', text)
    text = re.sub(r'[*_~`^|<>\\]+', '', text)
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Loading evaluator.

rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")

def compute_metrics(refs, preds) -> dict:
    results = {
        "ROUGE": rouge.compute(predictions=preds, references=refs),
        "BLEU": bleu.compute(predictions=preds, references=refs),
        "METEOR": meteor.compute(predictions=preds, references=refs)
    }
    return results


more_context, overall_metrics, all_preds, all_refs = [], [], [], []

for id, q in enumerate(questions):
    print(f"Computing question {id}")
    query_vec = embedding_model.embed_query(q)
    context_docs = db.similarity_search_by_vector(query_vec, k=K)
    context = "\n".join([d.page_content for d in context_docs])

    if id > 0:
        start = max(0, id - 3)
        prev_qa_pairs = []
        for j in range(start, id):
            prev_q = questions[j]
            prev_a = more_context[j]
            prev_qa_pairs.append(f"Q: {prev_q}\nA: {prev_a}")
        prev_context = "\n".join(prev_qa_pairs)
    else:
        prev_context = ""

    prompt = f"""
    You are a helpful and precise medical study assistant.
    Use the following context to answer the user's question.
    Write a detailed paragraph-style answer (no bullet points).
    If the answer is not present, say "I don't know; please consult the textbook."

    Context:
    {context}

    Previous Answer:
    {prev_context}

    Question: {q}
    Answer:
    """

    response = pipe(prompt, max_new_tokens=300, temperature=0.7)
    test_ans = clean_text(response[0]["generated_text"])
    more_context.append(test_ans)

    preds = [test_ans]
    refs = [FAQ["Expected_answer"].iloc[id]]

    metrics = compute_metrics(refs, preds)
    overall_metrics.append(metrics)

    all_preds.append(test_ans)
    all_refs.append(refs[0])

final_metrics = compute_metrics(all_refs, all_preds)
print("Final Metrics:", final_metrics, "\n")
print("Overall metrics through the iterations:", overall_metrics, "\n")


os.makedirs("data/final", exist_ok=True)
# Save the generated answers.
FAQ["Generated_Answer"] = more_context
FAQ.to_csv("data/final/FAQ_responses.csv", index=False)

FAQ['Generated_Answer'] = FAQ['Generated_Answer'].apply(clean_text) #Clean of any weirdness 
FAQ.to_csv("data/final/FAQ_cleaned_responses.csv", index=False)

# Saving scores.
with open('data/final/final_model_finalscore.json', 'w') as f:
    json.dump(final_metrics, f, indent=4)
print("Final model score saved.")

with open('data/final/final_model_overallscore.json', 'w') as f:
    json.dump(overall_metrics, f, indent=4)
print("Overall model score saved.")