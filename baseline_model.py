import pandas as pd #type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline #type: ignore
from huggingface_hub import login #type: ignore
from dotenv import load_dotenv #type: ignore
import os


load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
login(HF_TOKEN)
print("Logged in.")

FAQ = pd.read_csv('data/FAQ_Test.csv')
# print(FAQ.head(), "\n\n")

model_name = "google/gemma-3-1b-it"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=500,
    temperature=0.7,
    do_sample=True
)

questions = FAQ["Question"].tolist()  

responses = pipe(
    questions,          
    max_new_tokens=300,
    temperature=0.7,
)

test_answers = [r[0]["generated_text"] for r in responses]
FAQ["Answer"] = test_answers
print(f"FAQ dataset updated.")
FAQ.to_csv('data/baseline/FAQ_responses.csv', index=False)

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
FAQ_Cleaned.to_csv("data/baseline/FAQ_cleaned_responses.csv", index=False)