# 🚀 MedBot Web Deployment Guide

## Quick Start (Local)

```bash
# Install web dependencies
pip install -r requirements_web.txt

# Run the web app
python app.py
```

Then open: http://localhost:5000

---

## Deploy to Render (Free)

### Step 1: Prepare Files
All files are ready! You have:
- `app.py` - Flask application
- `templates/index.html` - Frontend
- `requirements_web.txt` - Dependencies
- `baseline_lstm_model.pth` - Trained model
- `vocab.pkl` - Vocabulary

### Step 2: Create Render Account
1. Go to https://render.com
2. Sign up with GitHub
3. Connect your MarcusV210/MedBot repository

### Step 3: Create Web Service
1. Click "New +" → "Web Service"
2. Connect your GitHub repository
3. Select branch: `Anamay`
4. Configure:
   - **Name:** medbot-ai
   - **Root Directory:** MedBot
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements_web.txt`
   - **Start Command:** `gunicorn app:app`
   - **Instance Type:** Free

### Step 4: Deploy
1. Click "Create Web Service"
2. Wait 5-10 minutes for deployment
3. Your app will be live at: `https://medbot-ai.onrender.com`

---

## Deploy to Hugging Face Spaces (Free)

### Step 1: Create Space
1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Name: `medbot-ai`
4. SDK: Gradio or Streamlit
5. Make it Public

### Step 2: Upload Files
Upload these files to your Space:
- `app.py`
- `templates/index.html`
- `requirements_web.txt`
- `baseline_lstm_model.pth`
- `vocab.pkl`

### Step 3: Auto-Deploy
Hugging Face will automatically deploy your app!

---

## Deploy to Railway (Free)

### Step 1: Create Account
1. Go to https://railway.app
2. Sign up with GitHub

### Step 2: Deploy
1. Click "New Project"
2. Select "Deploy from GitHub repo"
3. Choose MarcusV210/MedBot
4. Select branch: Anamay
5. Railway auto-detects Python and deploys!

---

## Environment Variables (if needed)

```bash
FLASK_ENV=production
PORT=5000
```

---

## File Structure

```
MedBot/
├── app.py                      # Flask backend
├── templates/
│   └── index.html             # Frontend UI
├── requirements_web.txt        # Web dependencies
├── baseline_lstm_model.pth    # Trained model (85MB)
├── vocab.pkl                  # Vocabulary
└── DEPLOYMENT.md              # This file
```

---

## Troubleshooting

### "Model file too large"
Some platforms have file size limits. Solutions:
1. Use Git LFS for large files
2. Download model on startup from cloud storage
3. Use a platform with higher limits (Render, Railway)

### "Out of memory"
Free tiers have limited RAM. Solutions:
1. Use CPU-only inference (already configured)
2. Upgrade to paid tier
3. Optimize model loading

### "Slow responses"
Free tiers have limited resources. Solutions:
1. Add caching for common questions
2. Use smaller embedding model
3. Upgrade to paid tier

---

## Production Optimizations

### 1. Add Caching
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def generate_answers(question):
    # ... existing code
```

### 2. Use Gunicorn (Production Server)
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### 3. Add HTTPS
Most platforms (Render, Railway) provide HTTPS automatically

### 4. Add Rate Limiting
```python
from flask_limiter import Limiter

limiter = Limiter(app, default_limits=["100 per hour"])
```

---

## Cost Estimate

### Free Tier (Recommended for Demo)
- **Render:** Free (sleeps after 15 min inactivity)
- **Railway:** $5 credit/month free
- **Hugging Face:** Completely free

### Paid Tier (Production)
- **Render:** $7/month (always on)
- **Railway:** ~$5-10/month
- **AWS/GCP:** ~$10-20/month

---

## Support

For issues:
1. Check logs in platform dashboard
2. Test locally first: `python app.py`
3. Verify all files are uploaded
4. Check file sizes and memory limits

---

**Your MedBot is ready to deploy!** 🚀

Choose any platform above and follow the steps. Render is recommended for beginners.
