# 🚀 MedBot Deployment Guide

## 📋 **Deployment Options**

### **Option 1: Streamlit Cloud (Recommended - Easiest)**
### **Option 2: Render (Free Flask hosting)**
### **Option 3: Railway (Modern deployment)**
### **Option 4: Heroku (Traditional)**

---

## 🎯 **Option 1: Streamlit Cloud (EASIEST)**

### **Steps:**
1. **Push to GitHub** (already done ✅)
2. **Go to:** https://share.streamlit.io/
3. **Sign in** with your GitHub account
4. **Click "New app"**
5. **Select:**
   - Repository: `MarcusV210/MedBot`
   - Branch: `Anamay`
   - Main file path: `streamlit_app.py`
6. **Add secrets** (click "Advanced settings"):
   ```
   GITHUB_TOKEN = "your_github_token_here"
   ```
7. **Click "Deploy!"**

### **Result:**
- **URL:** `https://your-app-name.streamlit.app/`
- **Features:** Simplified interface, easy to use
- **Cost:** 100% FREE
- **Time:** 5-10 minutes

---

## 🎯 **Option 2: Render (FREE Flask Hosting)**

### **Steps:**
1. **Go to:** https://render.com/
2. **Sign up** with GitHub
3. **Click "New +" → "Web Service"**
4. **Connect repository:** `MarcusV210/MedBot`
5. **Configure:**
   - Name: `medbot`
   - Branch: `Anamay`
   - Build Command: `pip install -r requirements_web.txt`
   - Start Command: `python app.py`
6. **Add environment variable:**
   - Key: `GITHUB_TOKEN`
   - Value: `your_github_token_here`
7. **Click "Create Web Service"**

### **Result:**
- **URL:** `https://medbot.onrender.com/`
- **Features:** Full Flask app with all features
- **Cost:** FREE (with some limitations)
- **Time:** 10-15 minutes

---

## 🎯 **Option 3: Railway (MODERN)**

### **Steps:**
1. **Go to:** https://railway.app/
2. **Sign up** with GitHub
3. **Click "New Project" → "Deploy from GitHub repo"**
4. **Select:** `MarcusV210/MedBot`
5. **Railway auto-detects** the `railway.json` config
6. **Add environment variable:**
   - Key: `GITHUB_TOKEN`
   - Value: `your_github_token_here`
7. **Deploy automatically**

### **Result:**
- **URL:** `https://medbot-production.up.railway.app/`
- **Features:** Modern deployment, fast
- **Cost:** FREE tier available
- **Time:** 5-10 minutes

---

## 🎯 **Option 4: Heroku (TRADITIONAL)**

### **Steps:**
1. **Install Heroku CLI:** https://devcenter.heroku.com/articles/heroku-cli
2. **Login:** `heroku login`
3. **Create app:** `heroku create medbot-anamay`
4. **Set environment variable:**
   ```bash
   heroku config:set GITHUB_TOKEN=your_github_token_here
   ```
5. **Deploy:**
   ```bash
   git push heroku Anamay:main
   ```

### **Result:**
- **URL:** `https://medbot-anamay.herokuapp.com/`
- **Features:** Traditional, reliable
- **Cost:** FREE tier (limited hours)
- **Time:** 15-20 minutes

---

## 🔧 **Files Created for Deployment**

### **For All Platforms:**
- ✅ `requirements.txt` - Streamlit dependencies
- ✅ `requirements_web.txt` - Flask dependencies (existing)
- ✅ `app.py` - Modified for production (PORT environment variable)

### **Platform-Specific:**
- ✅ `streamlit_app.py` - Streamlit version of the app
- ✅ `render.yaml` - Render configuration
- ✅ `railway.json` - Railway configuration
- ✅ `Procfile` - Heroku configuration

---

## 🎯 **Recommended: Streamlit Cloud**

### **Why Streamlit Cloud?**
1. **Easiest deployment** - Just connect GitHub repo
2. **100% FREE** - No credit card required
3. **Automatic updates** - Redeploys when you push to GitHub
4. **Built for ML/AI** - Perfect for your medical AI system
5. **Great for demos** - Perfect for professor presentation

### **Quick Demo URL:**
Once deployed on Streamlit Cloud, you'll get a URL like:
```
https://medbot-medical-ai.streamlit.app/
```

### **Features Available:**
- ✅ RAG System (83.9% accuracy)
- ✅ GitHub AI backup (77.0% accuracy)
- ✅ Chat history
- ✅ Performance metrics
- ✅ Professional interface

---

## 🔑 **GitHub Token Setup**

### **Get Your Token:**
1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Select scopes: `repo` (or default)
4. Copy the token (starts with `github_pat_`)

### **Add to Deployment:**
- **Streamlit:** Add in "Advanced settings" → Secrets
- **Render:** Add in Environment Variables
- **Railway:** Add in Variables tab
- **Heroku:** Use `heroku config:set`

---

## 🎉 **After Deployment**

### **Test Your Deployed App:**
1. **Visit the URL** provided by the platform
2. **Ask a medical question:** "What causes diabetes?"
3. **Check responses** from RAG and GitHub AI
4. **Verify performance** matches local testing

### **Share with Professor:**
- **Demo URL:** Share the live deployment link
- **GitHub Repo:** https://github.com/MarcusV210/MedBot/tree/Anamay
- **Performance:** RAG 83.9%, GitHub AI 77.0%
- **Evaluation:** Semantic similarity against medical literature

---

## 🚨 **Troubleshooting**

### **Common Issues:**

**"GitHub token not configured"**
- Solution: Add GITHUB_TOKEN environment variable

**"Models loading slowly"**
- Solution: First load takes time, subsequent loads are cached

**"Port already in use"**
- Solution: Platform handles ports automatically in production

**"Memory issues"**
- Solution: Use Streamlit version (lighter) or upgrade plan

---

## 🏆 **Deployment Success Checklist**

✅ **App loads without errors**  
✅ **RAG system returns medical content**  
✅ **GitHub AI provides backup responses**  
✅ **Chat history works**  
✅ **Performance metrics display**  
✅ **Professional interface loads**  

**Your MedBot is now live and ready for presentation!** 🎉

---

## 📞 **Next Steps**

1. **Choose deployment platform** (Streamlit Cloud recommended)
2. **Follow the steps** for your chosen platform
3. **Test the deployed app** thoroughly
4. **Share the URL** with your professor
5. **Present with confidence** - you have a live, working system!

**Deployment Time:** 5-15 minutes depending on platform  
**Result:** Live medical AI system accessible worldwide  
**Perfect for:** Academic presentations and demonstrations