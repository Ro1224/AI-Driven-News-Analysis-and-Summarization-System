# AI-Driven News Summarizer, Translator & Chatbot

## Overview
This project is a web application that allows users to:
- **Summarize news articles** from a given URL using advanced NLP models (BART, mBART, IndicBART).
- **Translate summaries** into multiple languages (English, Hindi, French, German, Spanish, Tamil, Arabic, Chinese).
- **Evaluate article reliability** using a RoBERTa-based classifier.
- **Ask questions** about the summarized article using a Cohere-powered chatbot.

The app is built with **Flask** and leverages state-of-the-art transformer models for summarization, translation, and reliability prediction. The UI is modern and responsive, with templates for summary display, translation, and chatbot interaction.

---

## Features
- **News Summarization:** Extracts and summarizes the main content of news articles from URLs.
- **Multilingual Translation:** Supports translation of summaries into several major languages.
- **Reliability Prediction:** Classifies articles as reliable or unreliable using a fine-tuned RoBERTa model.
- **Chatbot Q&A:** Users can ask questions about the summarized article, answered by Cohere's large language model.
- **Beautiful UI:** Responsive, user-friendly interface with Bootstrap and custom CSS.

---

## Directory Structure
```
backend.py                # Main Flask backend (API, logic, routes)
chatbot.py                # Standalone CLI chatbot for article Q&A
templates/                # HTML templates (Jinja2)
    index.html            # Landing page
    summary.html          # Summary and translation page
    query.html            # Chatbot Q&A page
static/                   # Static assets (images, CSS)
    img1.jpg
index.html                # (Legacy/alternate) HTML page
flask_session/            # Flask session storage
*.ipynb                   # Jupyter notebooks for experiments
.gitignore, venv/         # Standard project files
```

---

## Setup Instructions

### 1. Clone the Repository
```
git clone <repo-url>
cd "Main proj - 2"
```

### 2. Create & Activate Virtual Environment (Recommended)
```
python -m venv venv
venv\Scripts\activate   # On Windows
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```
If `requirements.txt` is missing, install manually:
```
pip install flask flask-session transformers torch langdetect beautifulsoup4 requests cohere
```

### 4. Download Pretrained Models
The first run will automatically download required models (BART, mBART, RoBERTa, IndicBART) from HuggingFace.

### 5. Set Cohere API Key
Replace the placeholder API key in `backend.py` and `chatbot.py` with your own from [Cohere](https://dashboard.cohere.com/api-keys).

---

## Running the Application

### Web App (Flask)
```
python backend.py
```
Visit [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser.

### CLI Chatbot
```
python chatbot.py
```

---

## Usage
1. **Enter a news article URL** on the homepage.
2. **Choose a target language** for translation.
3. **(Optional) Ask a question** about the article.
4. View the **summary, translation, reliability score, and chatbot answer**.

---

## Notes
- The app requires a GPU for best performance, but will run on CPU (slower).
- For production, set `debug=False` in `backend.py`.
- The app uses session storage for user data between routes.

---

## License
This project is for educational and research purposes only. See `LICENSE` for details.
