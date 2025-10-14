#!/usr/bin/env python3
"""Test the app integration without starting the full Flask server"""

import sys
import os

# Mock Flask components for testing
class MockSession(dict):
    modified = False

class MockRequest:
    @staticmethod
    def get_json():
        return {'question': 'What causes diabetes?'}

# Setup mocks
import flask
original_session = flask.session if hasattr(flask, 'session') else None

# Import after mocking
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("Testing MedBot Integration")
print("=" * 60)

# Test the GitHub API function directly
print("\n1. Testing GitHub Models API function...")
from app import call_github_models

messages = [
    {"role": "system", "content": "You are a medical AI assistant."},
    {"role": "user", "content": "What causes diabetes?"}
]

result = call_github_models(messages)

if result:
    print(f"✅ GitHub Models API working!")
    print(f"Response length: {len(result)} chars")
    print(f"Preview: {result[:200]}...")
else:
    print("❌ GitHub Models API failed")

print("\n" + "=" * 60)
print("✅ Integration test complete!")
print("=" * 60)
print("\nYou can now run: python app.py")
print("Then visit: http://localhost:5000")
