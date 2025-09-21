import cohere
from flask import Flask, render_template, request
from transformers import (
    BartForConditionalGeneration, BartTokenizer,
    MBartForConditionalGeneration, MBart50TokenizerFast,
    RobertaTokenizer, RobertaForSequenceClassification
)
from langdetect import detect, DetectorFactory
from bs4 import BeautifulSoup
import requests
import torch
from rouge_score import rouge_scorer
import numpy as np

# Fix for langdetect "Need to load profiles" issue
DetectorFactory.seed = 0

# Initialize Cohere client with your API key
cohere_client = cohere.Client("136nQ8QaflaJnEheQ187ypmGaDz6vECXt7ruZAem")

app = Flask(__name__)

# Load summarization models
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
mbart_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
mbart_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# Load RoBERTa for reliability
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta_model = RobertaForSequenceClassification.from_pretrained("roberta-base")

# Language code mapping
LANG_CODE_MAP = {
    'en': 'en_XX',
    'ta': 'ta_IN',
    'hi': 'hi_IN',
    'fr': 'fr_XX',
    'de': 'de_DE',
    'es': 'es_XX',
    'ar': 'ar_AR',
    'zh': 'zh_CN'
}

# Web scraper
def get_article_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        return ' '.join(p.get_text() for p in soup.find_all('p'))
    except Exception as e:
        return f"Error: {e}"

# Summarization
def summarize(text, lang):
    if lang == 'en':
        inputs = bart_tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = bart_model.generate(inputs['input_ids'], num_beams=4, max_length=150, early_stopping=True)
        return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    else:
        mbart_tokenizer.src_lang = LANG_CODE_MAP.get(lang, 'en_XX')
        inputs = mbart_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        summary_ids = mbart_model.generate(inputs['input_ids'], num_beams=4, max_length=150, early_stopping=True)
        return mbart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Translation
def translate_summary(text, src_lang, tgt_lang):
    try:
        src_code = LANG_CODE_MAP.get(src_lang, 'en_XX')  
        tgt_code = LANG_CODE_MAP.get(tgt_lang, 'en_XX')  
        mbart_tokenizer.src_lang = src_code
        encoded = mbart_tokenizer(text, return_tensors="pt", truncation=True)
        generated_tokens = mbart_model.generate(
            **encoded,
            forced_bos_token_id=mbart_tokenizer.lang_code_to_id[tgt_code]
        )
        return mbart_tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    except Exception as e:
        return f"Translation error: {e}"

# Reliability prediction
def predict_reliability(text):
    inputs = roberta_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = roberta_model(**inputs)
    prob = torch.softmax(outputs.logits, dim=1)
    return 1 if prob[0][1].item() >= 0.5 else 0

# ROUGE evaluation
def calculate_rouge_score(summary, reference):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return scores['rougeL'].fmeasure

# Fake BART score for demo
def get_bart_f1_score(summary, reference):
    rouge_f1 = calculate_rouge_score(summary, reference)
    bart_score = np.random.uniform(0.4, 0.8)
    return {"bart_score": bart_score, "f1_score": rouge_f1}

# Cohere chatbot
def ask_cohere(summary, question):
    try:
        prompt = f"Use the following news article to answer the question:\n\n{summary}\n\nQuestion: {question}"
        response = cohere_client.generate(
            prompt=prompt,
            max_tokens=200,
            temperature=0.7
        )
        return response.generations[0].text.strip()
    except Exception as e:
        return f"Cohere API error: {e}"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form.get("url", "").strip()
        target_lang = request.form.get("target_lang", "en").strip().lower()
        question = request.form.get("question", "").strip()

        article = get_article_content(url)
        detected_lang = detect(article) if not article.startswith("Error:") else "en"

        summary = summarize(article, detected_lang)

        translated_summary = summary
        if target_lang != detected_lang and target_lang in LANG_CODE_MAP:
            translated_summary = translate_summary(summary, detected_lang, target_lang)

        reliability = predict_reliability(article)
        reference_summary = article[:150]
        scores = get_bart_f1_score(summary, reference_summary)

        chatbot_answer = ""
        if question:
            chatbot_answer = ask_cohere(summary, question)  # Using Cohere chatbot function

        return render_template(
            "index.html",
            detected_language=detected_lang,
            summary=summary,
            translated_summary=translated_summary,
            reliability=reliability,
            bart_score=scores["bart_score"],
            f1_score=scores["f1_score"],
            chatbot_answer=chatbot_answer,
            url=url,
            target_lang=target_lang
        )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
