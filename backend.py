import cohere
from flask import Flask, render_template, request, redirect, url_for, session
from flask_session import Session
from transformers import (
    BartForConditionalGeneration, BartTokenizer,
    MBartForConditionalGeneration, MBart50TokenizerFast,
    RobertaTokenizer, RobertaForSequenceClassification,
    AutoModelForSeq2SeqLM, AutoTokenizer
)
from langdetect import detect, DetectorFactory
from bs4 import BeautifulSoup
import requests
import torch
import logging

# Setup
DetectorFactory.seed = 0
app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = False
Session(app)

logging.basicConfig(level=logging.DEBUG)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cohere_client = cohere.Client("136nQ8QaflaJnEheQ187ypmGaDz6vECXt7ruZAem")  

# Load models
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to(device)
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

mbart_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to(device)
mbart_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

roberta_model = RobertaForSequenceClassification.from_pretrained("roberta-base").to(device)
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

indicbart_model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indicbart").to(device)
indicbart_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indicbart")

LANG_CODE_MAP = {
    'en': 'en_XX', 'ta': 'ta_IN', 'hi': 'hi_IN',
    'fr': 'fr_XX', 'de': 'de_DE', 'es': 'es_XX',
    'ar': 'ar_AR', 'zh': 'zh_CN'
}


def get_article_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        article_text = ' '.join(p.get_text() for p in soup.find_all('p'))
        if len(article_text) < 100:
            return "Error: Article is too short or not fetched properly."
        return article_text
    except Exception as e:
        logging.error(f"Error fetching article: {e}")
        return f"Error: {e}"


def summarize(text, lang):
    try:
        logging.debug(f"Summarizing article. Language detected: {lang}")
        if lang == 'en':
            inputs = bart_tokenizer(text, return_tensors="pt", max_length=1024, truncation=True).to(device)
            summary_ids = bart_model.generate(inputs['input_ids'], num_beams=4, max_length=150, early_stopping=True)
            return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        elif lang == 'ta':
            inputs = indicbart_tokenizer(text, return_tensors="pt", max_length=1024, truncation=True).to(device)
            summary_ids = indicbart_model.generate(inputs['input_ids'], num_beams=4, max_length=150, early_stopping=True)
            return indicbart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        else:
            mbart_tokenizer.src_lang = LANG_CODE_MAP.get(lang, 'en_XX')
            inputs = mbart_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
            summary_ids = mbart_model.generate(inputs['input_ids'], num_beams=4, max_length=150, early_stopping=True)
            return mbart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    except Exception as e:
        logging.error(f"Error summarizing: {e}")
        return f"Error: {e}"


def translate_summary(text, src_lang, tgt_lang):
    try:
        src_code = LANG_CODE_MAP.get(src_lang, 'en_XX')
        tgt_code = LANG_CODE_MAP.get(tgt_lang, 'en_XX')
        mbart_tokenizer.src_lang = src_code
        encoded = mbart_tokenizer(text, return_tensors="pt", truncation=True).to(device)
        generated_tokens = mbart_model.generate(
            **encoded,
            forced_bos_token_id=mbart_tokenizer.lang_code_to_id[tgt_code]
        )
        return mbart_tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    except Exception as e:
        logging.error(f"Translation error: {e}")
        return f"Translation error: {e}"


def predict_reliability(text):
    try:
        inputs = roberta_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        outputs = roberta_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        return 1 if probs[0][1].item() >= 0.5 else 0
    except Exception as e:
        logging.error(f"Error predicting reliability: {e}")
        return "Error"


def ask_cohere(summary, question):
    try:
        prompt = f"Use the following news article to answer the question:\n\n{summary}\n\nQuestion: {question}"
        response = cohere_client.generate(prompt=prompt, max_tokens=200, temperature=0.7)
        return response.generations[0].text.strip()
    except Exception as e:
        logging.error(f"Cohere API error: {e}")
        return f"Cohere API error: {e}"


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form.get("url", "").strip()
        article = get_article_content(url)

        if article.startswith("Error:"):
            return render_template("error.html", error_message=article)

        detected_lang = detect(article)
        summary = summarize(article, detected_lang)

        if summary.startswith("Error:"):
            return render_template("error.html", error_message=summary)

        reliability = predict_reliability(article)

        session["url"] = url
        session["article"] = article
        session["summary"] = summary
        session["detected_lang"] = detected_lang
        session["reliability"] = reliability
        session.modified = True

        return redirect(url_for("summarize_route"))

    return render_template("index.html")


@app.route("/summarize", methods=["GET"])
def summarize_route():
    return render_template(
        "summary.html",
        summary=session.get("summary", ""),
        reliability=session.get("reliability", ""),
        url=session.get("url", "")
    )


@app.route("/translate", methods=["POST"])
def translate_route():
    target_lang = request.form.get("language", "en").lower()
    detected_lang = session.get("detected_lang", "en")
    summary = session.get("summary", "")

    if target_lang != detected_lang:
        translated_summary = translate_summary(summary, detected_lang, target_lang)
        session["summary"] = translated_summary
        session["detected_lang"] = target_lang
        session.modified = True
    else:
        translated_summary = summary

    return render_template(
        "summary.html",
        summary=translated_summary,
        reliability=session.get("reliability", ""),
        url=session.get("url", "")
    )


@app.route("/chatbot", methods=["GET", "POST"])
def chatbot_route():
    question = request.form.get("question", "") if request.method == "POST" else request.args.get("question", "")
    summary = session.get("summary", "")

    if question:
        answer = ask_cohere(summary, question)
        return render_template("query.html", question=question, answer=answer, url=session.get("url"))
    else:
        return render_template("query.html", question=None, answer="Please enter a question.", url=session.get("url"))


if __name__ == "__main__":
    app.run(debug=True)
