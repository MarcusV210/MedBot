#!/usr/bin/env python3
"""
One-Click MedBot Deployment Helper
This script will guide you through the deployment process
"""

import webbrowser
import time
import os

def main():
    print("🚀 MedBot One-Click Deployment Helper")
    print("=" * 50)
    print()
    
    print("📋 What this script will do:")
    print("1. Open Streamlit Cloud in your browser")
    print("2. Guide you through the deployment process")
    print("3. Provide your GitHub token for easy copy-paste")
    print()
    
    # Get GitHub token
    github_token = "your_github_token_here"
    
    print("🔑 Your GitHub Token (copy this):")
    print(f"GITHUB_TOKEN = \"{github_token}\"")
    print()
    
    input("Press ENTER to open Streamlit Cloud...")
    
    # Open Streamlit Cloud
    webbrowser.open("https://share.streamlit.io/")
    
    print("🌐 Streamlit Cloud opened in your browser!")
    print()
    print("📝 Follow these steps in the browser:")
    print("1. Click 'Sign in' → 'Continue with GitHub'")
    print("2. Click 'New app'")
    print("3. Fill in:")
    print("   Repository: MarcusV210/MedBot")
    print("   Branch: Anamay")
    print("   Main file path: streamlit_app.py")
    print("4. Click 'Advanced settings...'")
    print("5. In Secrets section, paste:")
    print(f"   GITHUB_TOKEN = \"{github_token}\"")
    print("6. Click 'Deploy!'")
    print()
    print("⏱️  Deployment takes 2-3 minutes")
    print("🎉 You'll get a URL like: https://medbot-anamay.streamlit.app/")
    print()
    
    input("Press ENTER when deployment is complete...")
    
    print("🎯 Test your deployed app:")
    print("1. Ask: 'What causes diabetes?'")
    print("2. See RAG (83.9%) + GitHub AI (77.0%) responses")
    print("3. Try follow-up: 'What are the treatment options?'")
    print()
    print("✅ Your MedBot is now live and ready for presentation!")

if __name__ == "__main__":
    main()