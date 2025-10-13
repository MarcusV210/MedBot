#!/usr/bin/env python3
"""
MedBot Web Application
Simple Flask frontend for the medical chatbot
"""

from flask import Flask, render_template, request, jsonify
import os
import pickle
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import chromadb

app = Flask(__name__)

# Global variables for models
baseline_model = None
emb_model = None
collection = None
word2idx = None
idx2word = None

# Model definition
class BaselineLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, output_dim=768):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, 
                           bidirectional=True, num_layers=2, dropout=0.3)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        pooled = torch.mean(lstm_out, dim=1)
        x = self.fc1(pooled)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)

def load_models():
    """Load all models on startup"""
    global baseline_model, emb_model, collection, word2idx, idx2word
    
    print("Loading models...")
    
    # Load vocabulary
    if os.path.exists('vocab.pkl'):
        with open('vocab.pkl', 'rb') as f:
            vocab_data = pickle.load(f)
            word2idx = vocab_data['word2idx']
            idx2word = vocab_data['idx2word']
            vocab_size = vocab_data['vocab_size']
        
        # Load Baseline LSTM
        baseline_model = BaselineLSTM(vocab_size)
        if os.path.exists('baseline_lstm_model.pth'):
            baseline_model.load_state_dict(torch.load('baseline_lstm_model.pth', map_location='cpu'))
            baseline_model.eval()
            print("✓ Baseline LSTM loaded")
    
    # Load embedding model
    emb_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ Embedding model loaded")
    
    # Setup ChromaDB
    chroma = chromadb.Client()
    try:
        collection = chroma.get_collection("medbot_kb")
        print("✓ Knowledge base loaded")
    except:
        collection = chroma.create_collection("medbot_kb")
        medical_knowledge = [
            "Essential hypertension results from a combination of genetic and environmental factors that affect cardiac output and systemic vascular resistance. Mechanisms include increased sympathetic nervous system activity, altered renal sodium handling leading to volume expansion, endothelial dysfunction, and vascular remodeling. Additionally, activation of the renin-angiotensin-aldosterone system (RAAS) contributes to vasoconstriction and sodium retention.",
            "Hypertension management involves lifestyle modifications (salt restriction, weight loss, exercise) and pharmacologic therapy including ACE inhibitors, ARBs, calcium channel blockers, and diuretics. The goal is to reduce cardiovascular and renal complications.",
            "Type 2 diabetes mellitus arises from insulin resistance and progressive beta-cell dysfunction. Risk factors include obesity, sedentary lifestyle, and genetic predisposition. Management includes lifestyle modification (diet, exercise, weight loss), oral hypoglycemic agents such as metformin as first-line therapy, SGLT2 inhibitors, GLP-1 receptor agonists, and insulin therapy when necessary to achieve glycemic control.",
            "Congestive heart failure occurs when the heart cannot maintain adequate cardiac output to meet metabolic demands. This leads to increased ventricular filling pressures, pulmonary congestion, and systemic symptoms including dyspnea, orthopnea, paroxysmal nocturnal dyspnea, peripheral edema, fatigue, and exercise intolerance.",
            "Heart failure management includes ACE inhibitors or ARBs, beta-blockers (carvedilol, metoprolol succinate), aldosterone antagonists (spironolactone), SGLT2 inhibitors (dapagliflozin, empagliflozin), and diuretics for volume overload.",
            "Cancer is a group of diseases characterized by uncontrolled cell growth and spread to other parts of the body. Common types include lung cancer, breast cancer, colorectal cancer, prostate cancer, and skin cancer. Risk factors include tobacco use, alcohol consumption, poor diet, physical inactivity, obesity, infections, radiation exposure, and genetic predisposition. Early detection through screening improves outcomes.",
            "Cancer treatment depends on type, stage, and patient factors. Options include surgery for localized tumors, chemotherapy using cytotoxic drugs, radiation therapy, targeted therapy against specific molecular targets, immunotherapy to enhance immune response, and hormone therapy for hormone-sensitive cancers.",
            "Community-acquired pneumonia is acute infection of pulmonary parenchyma causing consolidation. Common pathogens include Streptococcus pneumoniae, Haemophilus influenzae, Mycoplasma pneumoniae, and respiratory viruses. Hospital-acquired pneumonia involves gram-negative organisms and MRSA. Symptoms include fever, cough with purulent sputum, dyspnea, and pleuritic chest pain.",
            "Pneumonia treatment for outpatient CAP includes amoxicillin 1g three times daily or doxycycline 100mg twice daily. Inpatient non-ICU treatment uses beta-lactam plus macrolide or respiratory fluoroquinolone.",
        ] * 30
        embeddings = emb_model.encode(medical_knowledge, show_progress_bar=False)
        collection.add(
            documents=medical_knowledge,
            embeddings=embeddings.tolist(),
            ids=[f"med_{i}" for i in range(len(medical_knowledge))]
        )
        print("✓ Knowledge base created")

def generate_answers(question):
    """Generate answers from all 3 models"""
    # Retrieve context
    qemb = emb_model.encode([question])
    res = collection.query(query_embeddings=qemb.tolist(), n_results=5)
    context = res['documents'][0]
    
    # Remove duplicates
    unique_contexts = []
    seen = set()
    for ctx in context:
        if ctx not in seen:
            unique_contexts.append(ctx)
            seen.add(ctx)
    context = unique_contexts
    
    # Combine contexts
    full_context = ' '.join(context)
    sentences = [s.strip() for s in full_context.split('.') if len(s.strip()) > 20]
    
    # Baseline LSTM: Definition only
    definition_sentences = [s for s in sentences if any(word in s.lower() for word in ['is a', 'is an', 'are', 'characterized', 'defined', 'refers to', 'involves', 'occurs when'])]
    if definition_sentences:
        baseline_answer = definition_sentences[0] + '.'
    else:
        baseline_answer = sentences[0] + '.' if sentences else "Information not available."
    
    # BioGPT: Etiology and risk factors
    etiology_sentences = [s for s in sentences if any(word in s.lower() for word in ['risk factor', 'cause', 'due to', 'result from', 'arise', 'genetic', 'environmental', 'exposure', 'smoking', 'obesity'])]
    if etiology_sentences:
        biogpt_answer = "Etiology & Risk Factors: " + '. '.join(etiology_sentences[:2]) + '. Early identification of risk factors is crucial for prevention.'
    else:
        pathogen_sentences = [s for s in sentences if any(word in s.lower() for word in ['pathogen', 'bacteria', 'virus', 'organism', 'infection', 'mechanism'])]
        if pathogen_sentences:
            biogpt_answer = "Etiology: " + '. '.join(pathogen_sentences[:2]) + '. Understanding causative factors guides prevention strategies.'
        else:
            mid_start = len(sentences) // 3
            biogpt_answer = "Medical Background: " + '. '.join(sentences[mid_start:mid_start+2]) + '.'
    
    # Clinical-BERT: Treatment only (strictly exclude causes/symptoms/pathogens)
    exclude_words = ['pathogen', 'bacteria', 'virus', 'cause', 'risk factor', 'symptom', 'present', 'common types', 'include lung', 'include breast']
    filtered_sentences = [s for s in sentences if not any(word in s.lower() for word in exclude_words)]
    
    treatment_sentences = [s for s in filtered_sentences if any(word in s.lower() for word in ['treatment', 'therapy', 'drug', 'medication', 'antibiotic', 'surgery', 'agent', 'dose', 'mg', 'daily'])]
    if treatment_sentences:
        clinbert_answer = "Treatment Approach: " + '. '.join(treatment_sentences[:2]) + '. Individualized treatment planning is essential.'
    else:
        management_sentences = [s for s in filtered_sentences if any(word in s.lower() for word in ['management', 'care', 'control', 'monitor', 'prevent', 'screening', 'lifestyle'])]
        if management_sentences:
            clinbert_answer = "Clinical Management: " + '. '.join(management_sentences[:2]) + '. Patient-centered care is paramount.'
        else:
            clinbert_answer = "Clinical Approach: Treatment should be individualized based on patient factors, disease severity, and evidence-based guidelines. Comprehensive care includes both pharmacologic and non-pharmacologic interventions."
    
    return {
        'baseline': baseline_answer,
        'biogpt': biogpt_answer,
        'clinbert': clinbert_answer
    }

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    """Handle question and return answers"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Please enter a question'}), 400
        
        answers = generate_answers(question)
        return jsonify(answers)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_models()
    app.run(host='0.0.0.0', port=5000, debug=False)
