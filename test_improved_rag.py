#!/usr/bin/env python3
"""
Test the improved RAG system
"""

import requests
import json

def test_rag_question(question):
    """Test a single question"""
    print(f"\n🔄 Testing: {question}")
    
    try:
        response = requests.post('http://localhost:5000/ask', 
                               json={'question': question}, 
                               timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            rag_answer = data.get('rag', 'No RAG answer')
            
            print(f"✅ RAG Answer ({len(rag_answer)} chars):")
            print(f"   {rag_answer[:200]}...")
            
            return len(rag_answer), rag_answer
        else:
            print(f"❌ Error: {response.status_code}")
            return 0, ""
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 0, ""

def main():
    print("=" * 60)
    print("TESTING IMPROVED RAG SYSTEM")
    print("=" * 60)
    
    # Test questions that should have good RAG coverage
    test_questions = [
        "What causes hypertension?",
        "What causes heart failure?", 
        "What causes diabetes?",
        "What causes iron deficiency anemia?",
        "What causes chronic kidney disease?"
    ]
    
    total_length = 0
    for question in test_questions:
        length, answer = test_rag_question(question)
        total_length += length
    
    print(f"\n" + "=" * 60)
    print(f"RESULTS:")
    print(f"Average RAG answer length: {total_length / len(test_questions):.0f} characters")
    print(f"Expected: 400-800 characters (comprehensive textbook content)")
    
    if total_length / len(test_questions) > 300:
        print("✅ RAG system is now returning comprehensive content!")
    else:
        print("⚠️  RAG answers still seem short")
    
    print("=" * 60)

if __name__ == "__main__":
    main()