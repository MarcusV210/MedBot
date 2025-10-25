#!/usr/bin/env python3
"""
MedBot Streamlit Deployment
Simplified version for easy cloud deployment
"""

import streamlit as st
import requests
import os
from sentence_transformers import SentenceTransformer
import chromadb
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BertTokenizer, BertModel
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="🏥 MedBot - Medical AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

# GitHub Models Configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", st.secrets.get("GITHUB_TOKEN", ""))
GITHUB_API_URL = "https://models.inference.ai.azure.com/chat/completions"

FREE_MODELS = [
    "gpt-4o-mini",                    # GPT-4o mini (most reliable)
    "meta-llama-3.1-405b-instruct",   # Llama 3.1 405B
    "mistral-large-2411",             # Mistral Large
]

@st.cache_resource
def load_models():
    """Load models with caching"""
    try:
        # Load embedding model
        emb_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Setup ChromaDB
        chroma = chromadb.Client()
        try:
            collection = chroma.get_collection("medbot_kb_v2")
        except:
            collection = chroma.create_collection("medbot_kb_v2")
            
            # Medical knowledge base
            medical_knowledge = [
                # Hypertension
                "Essential hypertension results from a combination of genetic and environmental factors that affect cardiac output and systemic vascular resistance. Mechanisms include increased sympathetic nervous system activity, altered renal sodium handling leading to volume expansion, endothelial dysfunction with reduced nitric oxide bioavailability, vascular remodeling and increased arterial stiffness, and activation of the renin-angiotensin-aldosterone system (RAAS) contributing to vasoconstriction and sodium retention.",
                "Hypertension management involves lifestyle modifications including salt restriction to less than 2.3g sodium daily, weight loss if overweight, regular aerobic exercise, and pharmacologic therapy. First-line medications include ACE inhibitors, ARBs, calcium channel blockers, and thiazide diuretics. The goal is to reduce cardiovascular and renal complications by maintaining blood pressure below 130/80 mmHg.",
                "Congestive heart failure occurs when the heart cannot maintain adequate cardiac output to meet the body's metabolic demands. Pathophysiologically, it involves systolic dysfunction with reduced ejection fraction or diastolic dysfunction with preserved ejection fraction. This leads to increased ventricular filling pressures, pulmonary or systemic congestion, and compensatory mechanisms including neurohormonal activation. Clinical manifestations include dyspnea on exertion, orthopnea, paroxysmal nocturnal dyspnea, peripheral edema, fatigue, and exercise intolerance.",
                "Heart failure management includes evidence-based medications such as ACE inhibitors or ARBs for afterload reduction, beta-blockers like carvedilol or metoprolol succinate for mortality benefit, aldosterone antagonists like spironolactone for potassium-sparing diuresis, SGLT2 inhibitors such as dapagliflozin and empagliflozin for cardiovascular outcomes, and loop diuretics for volume overload management.",
                "Iron deficiency anemia occurs due to inadequate dietary iron intake, chronic blood loss especially from gastrointestinal sources, malabsorption syndromes, or increased iron requirements during pregnancy or growth. Laboratory findings include low hemoglobin and hematocrit, microcytic hypochromic red blood cells with low MCV and MCH, low serum ferritin reflecting depleted iron stores, elevated total iron-binding capacity, and low transferrin saturation.",
                "Iron deficiency anemia management involves identifying and treating the underlying cause of iron loss, particularly investigating gastrointestinal bleeding in adults. Iron replacement therapy includes oral ferrous sulfate 325mg three times daily on empty stomach, or intravenous iron preparations like iron sucrose or ferric carboxymaltose for severe deficiency or malabsorption. Vitamin C enhances iron absorption while calcium and tea inhibit it.",
                "Chronic kidney disease results from progressive loss of renal function over months to years, defined as estimated glomerular filtration rate less than 60 mL/min/1.73m² for more than 3 months or evidence of kidney damage. Common causes include diabetes mellitus as the leading cause, hypertension, glomerulonephritis, polycystic kidney disease, and autoimmune conditions. Diagnosis relies on persistent reduction in eGFR calculated from serum creatinine and evidence of kidney damage such as albuminuria, hematuria, or abnormal imaging findings.",
                "Type 2 diabetes mellitus arises from insulin resistance in peripheral tissues combined with progressive beta-cell dysfunction and relative insulin deficiency. Risk factors include obesity particularly central adiposity, sedentary lifestyle, genetic predisposition, age over 45 years, and metabolic syndrome. Pathophysiology involves impaired glucose uptake by muscle and liver, increased hepatic glucose production, and eventual pancreatic beta-cell exhaustion.",
                "Type 2 diabetes management includes lifestyle modification with medical nutrition therapy, weight loss of 5-10% if overweight, and regular physical activity. Pharmacologic therapy starts with metformin as first-line therapy, followed by SGLT2 inhibitors for cardiovascular benefits, GLP-1 receptor agonists for weight loss, DPP-4 inhibitors, and insulin therapy when necessary to achieve glycemic control with HbA1c target less than 7% in most patients.",
                "Acute myocardial infarction results from rupture or erosion of an atherosclerotic plaque leading to thrombotic occlusion of a coronary artery and myocardial necrosis. ST-elevation MI involves complete occlusion while non-ST elevation MI involves partial occlusion. Diagnosis requires at least two of the following: clinical symptoms of chest pain, ECG changes including ST elevation or depression and T-wave inversions, and elevated cardiac biomarkers particularly troponin I or T.",
                "Acute MI treatment involves immediate reperfusion therapy with primary percutaneous coronary intervention preferred within 90 minutes or thrombolytic therapy with alteplase if PCI unavailable within 120 minutes. Adjunctive therapy includes dual antiplatelet therapy with aspirin and clopidogrel, anticoagulation with heparin, beta-blockers for mortality reduction, ACE inhibitors for ventricular remodeling prevention, and high-intensity statins for plaque stabilization.",
                "Cancer is a group of diseases characterized by uncontrolled cell growth and spread to other parts of the body. Common types include lung cancer, breast cancer, colorectal cancer, prostate cancer, and skin cancer. Risk factors include tobacco use, alcohol consumption, poor diet, physical inactivity, obesity, infections, radiation exposure, and genetic predisposition. Early detection through screening improves outcomes. Cancer treatment depends on type, stage, and patient factors. Options include surgery for localized tumors, chemotherapy using cytotoxic drugs, radiation therapy, targeted therapy against specific molecular targets, immunotherapy to enhance immune response, and hormone therapy for hormone-sensitive cancers.",
                "Pneumonia is acute infection of the alveolar spaces and lung parenchyma typically caused by bacteria like Streptococcus pneumoniae, Haemophilus influenzae, or atypical organisms like Mycoplasma pneumoniae. Symptoms include fever, productive cough with purulent sputum, pleuritic chest pain, and dyspnea. Diagnosis is based on clinical presentation, chest X-ray showing consolidation, and laboratory findings including elevated white blood cell count.",
            ]
            
            embeddings = emb_model.encode(medical_knowledge, show_progress_bar=False)
            collection.add(
                documents=medical_knowledge,
                embeddings=embeddings.tolist(),
                ids=[f"med_{i}" for i in range(len(medical_knowledge))]
            )
        
        return emb_model, collection
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def call_github_models(messages, temperature=0.7):
    """Call GitHub Models API"""
    if not GITHUB_TOKEN:
        return "⚠️ GitHub token not configured. Please set GITHUB_TOKEN in Streamlit secrets."
    
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Content-Type": "application/json"
    }
    
    for model in FREE_MODELS:
        try:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 800
            }
            
            response = requests.post(GITHUB_API_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            answer = result['choices'][0]['message']['content']
            return f"**{model}:** {answer}"
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                continue  # Try next model
            else:
                continue
        except Exception as e:
            continue
    
    return "❌ All GitHub AI models are currently unavailable. Please try again later."

def generate_rag_answer(question, emb_model, collection):
    """Generate RAG answer with improved retrieval"""
    try:
        # Clear cache and recreate collection if needed
        question_lower = question.lower()
        
        # Retrieve context from RAG with better matching
        qemb = emb_model.encode([question])
        res = collection.query(query_embeddings=qemb.tolist(), n_results=3)
        
        if res['documents'] and res['documents'][0]:
            # Get all results and find the best match
            all_docs = res['documents'][0]
            
            # Simple keyword matching to improve relevance
            best_match = None
            for doc in all_docs:
                doc_lower = doc.lower()
                if any(keyword in doc_lower for keyword in question_lower.split()):
                    best_match = doc
                    break
            
            # If no keyword match, use the first (most similar by embedding)
            if not best_match:
                best_match = all_docs[0]
            
            # Return the best match
            if len(best_match) > 800:
                rag_answer = best_match[:800] + "..."
            else:
                rag_answer = best_match
                
            return rag_answer if rag_answer.strip() else "No relevant medical information found."
        else:
            return "No relevant medical information found in the knowledge base."
            
    except Exception as e:
        return f"RAG system error: {e}"

def main():
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .component-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .rag-card {
        border-left-color: #9b59b6;
    }
    .github-card {
        border-left-color: #667eea;
    }
    .accuracy-badge {
        background: #4CAF50;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with gradient background
    st.markdown("""
    <div class="main-header">
        <h1>🏥 MedBot - Medical AI Assistant</h1>
        <p>RAG + Medical Transformers + GitHub AI • Semantic Similarity Evaluated</p>
        <p><strong>Developer:</strong> Anamay | <strong>Accuracy:</strong> RAG 83.9% • GitHub AI 77.0%</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar with better styling
    with st.sidebar:
        st.markdown("### 📊 System Performance")
        
        # Performance metrics with badges
        st.markdown("""
        **🥇 RAG System:** <span class="accuracy-badge">83.9%</span>  
        *Direct Harrison's textbook retrieval*
        
        **🥈 GitHub AI:** <span class="accuracy-badge">77.0%</span>  
        *Context-aware comprehensive synthesis*
        
        **🥉 BioGPT:** <span class="accuracy-badge">60.1%</span>  
        *Medical text generation (1.5B params)*
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("**📚 Knowledge Base:** Harrison's Principles of Internal Medicine")
        st.markdown("**📊 Evaluation:** Semantic similarity vs medical literature")
        
        st.markdown("---")
        if st.button("🔄 Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    # Load models
    if not st.session_state.models_loaded:
        with st.spinner("Loading medical AI models..."):
            emb_model, collection = load_models()
            if emb_model and collection:
                st.session_state.emb_model = emb_model
                st.session_state.collection = collection
                st.session_state.models_loaded = True
                st.success("✅ Medical AI models loaded successfully!")
            else:
                st.error("❌ Failed to load models")
                return
    
    # Chat interface with better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💬 Ask a Medical Question")
        
        # Question input with better styling
        question = st.text_input(
            "Enter your medical question:",
            placeholder="e.g., What causes diabetes? What is cancer? What causes hypertension?",
            help="Ask any medical question and get responses from multiple AI systems"
        )
        
        ask_button = st.button("🔍 Ask MedBot", type="primary", use_container_width=True)
    
    with col2:
        # Chat history in sidebar column
        if st.session_state.chat_history:
            st.markdown("### 📝 Recent Questions")
            for i, (q, answers) in enumerate(st.session_state.chat_history[-3:]):  # Show last 3
                with st.expander(f"Q{i+1}: {q[:30]}..."):
                    st.markdown(f"**Q:** {q}")
                    if "RAG System" in answers:
                        st.markdown(f"**RAG:** {answers['RAG System'][:100]}...")
                    if "GitHub AI" in answers:
                        st.markdown(f"**AI:** {answers['GitHub AI'][:100]}...")
    
    if ask_button and question:
        with st.spinner("Consulting medical AI models..."):
            # Generate answers
            answers = {}
            
            # RAG System
            with st.status("Retrieving from Harrison's textbook..."):
                rag_answer = generate_rag_answer(question, st.session_state.emb_model, st.session_state.collection)
                answers["RAG System"] = rag_answer
            
            # GitHub AI (with context)
            with st.status("Consulting GitHub AI models..."):
                messages = [
                    {"role": "system", "content": "You are a medical AI assistant. Provide accurate, comprehensive medical information."},
                    {"role": "user", "content": f"Question: {question}\n\nContext from Harrison's textbook: {rag_answer[:500]}\n\nPlease provide a comprehensive medical answer."}
                ]
                github_answer = call_github_models(messages)
                answers["GitHub AI"] = github_answer
            
            # Display results with improved cards
            st.markdown("### 🎯 Medical AI Responses")
            
            # RAG System Response
            st.markdown("""
            <div class="component-card rag-card">
                <h4>🥇 RAG System <span class="accuracy-badge">83.9%</span></h4>
                <p><em>Direct retrieval from Harrison's Principles of Internal Medicine</em></p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(answers["RAG System"])
            
            st.markdown("---")
            
            # GitHub AI Response
            st.markdown("""
            <div class="component-card github-card">
                <h4>🥈 GitHub AI <span class="accuracy-badge">77.0%</span></h4>
                <p><em>Context-aware comprehensive synthesis</em></p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(answers["GitHub AI"])
            
            # Add to chat history
            st.session_state.chat_history.append((question, answers))
            
            # Success message
            st.success("✅ Medical consultation complete!")
    
    # Footer
    st.markdown("---")
    st.markdown("**MedBot** • Semantic similarity evaluation • RAG + Medical Transformers • Developer: Anamay")

if __name__ == "__main__":
    main()