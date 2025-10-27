import pandas as pd #type: ignore
import re
import evaluate #type: ignore
import json

test_set = pd.read_csv("data/baseline/FAQ_cleaned_responses.csv")

def clean(text: str) -> str:
    if not isinstance(text, str):
        return text
    
    text = re.sub(r'\\[nrt]', ' ', text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?;:'\"()\-\u2019]", "", text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()
test_set['Answer'] = test_set['Answer'].apply(clean)

rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")

def compute_metrics(df):
    refs = df["Expected_answer"].astype(str).tolist()
    preds = df["Answer"].astype(str).tolist()

    results = {
        "ROUGE": rouge.compute(predictions=preds, references=refs),
        "BLEU": bleu.compute(predictions=preds, references=refs),
        "METEOR": meteor.compute(predictions=preds, references=refs)
    }
    return results

results = compute_metrics(test_set)
print("Results calculated.")

with open('data/baseline/baseline_model_score.json', 'w') as f:
    json.dump(results, f, indent=4)

print("Files saved.")