#!/usr/bin/env python3
"""
Quick evaluation of improved RAG system
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests

def quick_evaluation():
    """Quick evaluation of top 5 FAQ questions"""
    
    # Load FAQ data
    df = pd.read_csv('FAQ_Test.csv')
    
    # Test first 5 questions
    test_questions = df.head(5)
    
    # Load similarity model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("=" * 80)
    print("QUICK RAG EVALUATION - IMPROVED SYSTEM")
    print("=" * 80)
    
    rag_scores = []
    
    for idx, row in test_questions.iterrows():
        question = row['Question']
        expected = row['Expected_answer']
        
        print(f"\n🔄 Testing Q{idx+1}: {question[:50]}...")
        
        try:
            response = requests.post('http://localhost:5000/ask', 
                                   json={'question': question}, 
                                   timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                rag_answer = data.get('rag', 'No answer')
                
                # Calculate semantic similarity
                embeddings = model.encode([expected, rag_answer])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                score = similarity * 100
                
                rag_scores.append(score)
                print(f"   RAG Score: {score:.1f}%")
                print(f"   Answer length: {len(rag_answer)} chars")
                
            else:
                print(f"   ❌ API Error: {response.status_code}")
                rag_scores.append(0)
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            rag_scores.append(0)
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY:")
    print("=" * 80)
    print(f"RAG Average Score: {np.mean(rag_scores):.1f}%")
    print(f"Individual Scores: {[f'{s:.1f}%' for s in rag_scores]}")
    
    if np.mean(rag_scores) > 70:
        print("🏆 EXCELLENT! RAG system now has high accuracy!")
    elif np.mean(rag_scores) > 50:
        print("✅ GOOD! RAG system significantly improved!")
    else:
        print("⚠️  Still needs improvement")
    
    print("=" * 80)
    
    return np.mean(rag_scores)

if __name__ == "__main__":
    quick_evaluation()