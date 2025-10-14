#!/usr/bin/env python3
"""Test GitHub Models API connection"""

import requests
import json
import os

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "your_github_token_here")  # Set via environment variable
GITHUB_API_URL = "https://models.inference.ai.azure.com/chat/completions"

FREE_MODELS = [
    "deepseek-r1",
    "gpt-4o-mini",
    "meta-llama-3.1-405b-instruct",
    "mistral-large-2411",
]

def test_model(model_name):
    """Test a specific model"""
    print(f"\n🔄 Testing {model_name}...")
    
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a medical AI assistant."},
            {"role": "user", "content": "What causes diabetes? Give a brief answer."}
        ],
        "temperature": 0.7,
        "max_tokens": 200
    }
    
    try:
        response = requests.post(GITHUB_API_URL, headers=headers, json=payload, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            answer = result['choices'][0]['message']['content']
            print(f"✅ SUCCESS!")
            print(f"Answer: {answer[:150]}...")
            return True
        else:
            print(f"❌ FAILED")
            print(f"Response: {response.text[:300]}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Testing GitHub Models API")
    print("=" * 60)
    
    success_count = 0
    for model in FREE_MODELS:
        if test_model(model):
            success_count += 1
            break  # Stop after first success
    
    print("\n" + "=" * 60)
    if success_count > 0:
        print("✅ GitHub Models API is working!")
    else:
        print("❌ All models failed. Check your token or GitHub Models access.")
    print("=" * 60)
