import pandas as pd
import numpy as np
import nltk
import re
import streamlit as st
import subprocess
import sys

# NLP
import spacy
from newspaper import Article
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords

# Visuals
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

import feedparser
from urllib.parse import quote

# -----------------------------
# SETUP (CLOUD SAFE)
# -----------------------------
nltk.download('stopwords', quiet=True)

# FIXED spaCy loading
try:
    nlp = spacy.load("en_core_web_sm")
except:
    subprocess.run(
        [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
        stdout=subprocess.DEVNULL
    )
    nlp = spacy.load("en_core_web_sm")

stop_words = set(stopwords.words('english'))

# -----------------------------
# FUNCTIONS
# -----------------------------
def fetch_news_urls(query, num_articles=5):
    query = quote(query)
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    
    feed = feedparser.parse(url)
    return [entry.link for entry in feed.entries[:num_articles]]

def extract_article(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except:
        return ""

def preprocess(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    doc = nlp(text)
    return [
        token.lemma_
        for token in doc
        if token.text not in stop_words and len(token.text) > 3
    ]

def prepare_corpus(urls):
    texts = []
    for url in urls:
        article = extract_article(url)
        tokens = preprocess(article)
        texts.append(tokens)

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    return texts, corpus, dictionary

def run_lda(corpus, dictionary, num_topics=4):
    return LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=10,
        random_state=42
    )

@st.cache_data
def load_models_dynamic(urls1, urls2):
    p_texts, p_corpus, p_dict = prepare_corpus(urls1)
    y_texts, y_corpus, y_dict = prepare_corpus(urls2)

    lda_p = run_lda(p_corpus, p_dict)
    lda_y = run_lda(y_corpus, y_dict)

    return p_texts, y_texts, lda_p, lda_y

# -----------------------------
# UI CONFIG
# -----------------------------
st.set_page_config(page_title="Political Topic Analysis", layout="wide")

st.title("🧠 Political Speech Topic Analysis Dashboard")

# -----------------------------
# INPUT
# -----------------------------
query1 = st.text_input("Enter Politician 1", "Pinarayi Vijayan")
query2 = st.text_input("Enter Politician 2", "Yogi Adityanath")

if st.button("Fetch & Analyze", key="fetch_main"):
    with st.spinner("Fetching articles and running analysis..."):
        urls1 = fetch_news_urls(query1)
        urls2 = fetch_news_urls(query2)

        p_texts, y_texts, lda_p, lda_y = load_models_dynamic(urls1, urls2)

        st.session_state["data"] = (p_texts, y_texts, lda_p, lda_y)

# -----------------------------
# SESSION CHECK
# -----------------------------
if "data" not in st.session_state:
    st.warning("Click 'Fetch & Analyze' to load data")
    st.stop()

p_texts, y_texts, lda_p, lda_y = st.session_state["data"]

# -----------------------------
# VISUAL FUNCTIONS
# -----------------------------
def generate_wordcloud(texts, title):
    all_words = " ".join([" ".join(text) for text in texts])
    wc = WordCloud(width=800, height=400).generate(all_words)

    fig, ax = plt.subplots()
    ax.imshow(wc)
    ax.axis("off")
    ax.set_title(title)
    st.pyplot(fig)

def show_topic_bar(model, title):
    topics = model.show_topics(num_topics=4, formatted=False)

    words, weights = [], []
    for topic in topics:
        for word, weight in topic[1]:
            words.append(word)
            weights.append(weight)

    fig, ax = plt.subplots()
    ax.barh(words, weights)
    ax.set_title(title)
    st.pyplot(fig)

def venn_diagram(p_texts, y_texts):
    p_words = set(word for text in p_texts for word in text)
    y_words = set(word for text in y_texts for word in text)

    fig, ax = plt.subplots()
    venn2([p_words, y_words], set_labels=("Politician 1", "Politician 2"))
    st.pyplot(fig)

# -----------------------------
# UI SECTIONS
# -----------------------------
section = st.sidebar.radio("Select View", [
    "Topics",
    "Word Clouds",
    "Topic Importance",
    "Comparison"
])

if section == "Topics":
    option = st.selectbox("Select Politician", ["Politician 1", "Politician 2"])

    def show_topics(model):
        for i, topic in model.print_topics(-1):
            st.write(f"### Topic {i}")
            st.write(topic)

    show_topics(lda_p if option == "Politician 1" else lda_y)

elif section == "Word Clouds":
    col1, col2 = st.columns(2)
    with col1:
        generate_wordcloud(p_texts, "Politician 1")
    with col2:
        generate_wordcloud(y_texts, "Politician 2")

elif section == "Topic Importance":
    col1, col2 = st.columns(2)
    with col1:
        show_topic_bar(lda_p, "Politician 1")
    with col2:
        show_topic_bar(lda_y, "Politician 2")

elif section == "Comparison":
    st.subheader("Vocabulary Overlap")
    venn_diagram(p_texts, y_texts)
