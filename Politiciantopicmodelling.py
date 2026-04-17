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

# -----------------------------
# SETUP (CLOUD SAFE)
# -----------------------------
nltk.download('stopwords', quiet=True)

@st.cache_resource
def load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except:
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            stdout=subprocess.DEVNULL
        )
        return spacy.load("en_core_web_sm")

nlp = load_spacy()

stop_words = set(stopwords.words('english'))

# -----------------------------
# DATA
# -----------------------------
PINARAYI_URLS = [
    "https://timesofindia.indiatimes.com/city/thiruvananthapuram/student-death-raises-caste-bias-concerns-says-cm-pinarayi-vijayan/articleshow/130267482.cms",
    "https://www.ndtv.com/india-news/if-up-turns-into-kerala-pinarayi-vijayan-sneers-at-yogi-adityanath-2760326",
    "https://www.ndtv.com/india-news/pinarayi-vijayan-inappropriate-kerala-chief-minister-on-yogi-adityanaths-up-kerala-remark-2781944",
    "https://timesofindia.indiatimes.com/city/thiruvananthapuram/pinarayi-vijayan-slams-adityanath-over-his-remarks-on-kerala/articleshow/60960704.cms"
]

YOGI_URLS = [
    "https://indiatimes.com/trending/nations-safety-lies-in-eliminating-the-wicked-uttar-pradesh-chief-minister-yogi-adityanath-669206.html",
    "https://indiatimes.com/trending/ramayana-and-mahabharatas-villains-reappear-in-modern-times-says-cm-yogi-during-lord-rams-rajtilak-672556.html",
    "https://timesofindia.indiatimes.com/city/lucknow/pappu-tappu-appu-of-india-bloc-cant-see-development-under-pm-modi-yogi/articleshow/125061291.cms",
    "https://indianexpress.com/article/india/up-cm-yogi-adityanath-uses-love-jihad-to-target-left-in-kerala-4874887/"
]

# -----------------------------
# FUNCTIONS
# -----------------------------
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
    tokens = [
        token.lemma_
        for token in doc
        if token.text not in stop_words and len(token.text) > 3
    ]
    return tokens

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

# -----------------------------
# CACHE (IMPORTANT)
# -----------------------------
@st.cache_resource
def load_models():
    p_texts, p_corpus, p_dict = prepare_corpus(PINARAYI_URLS)
    y_texts, y_corpus, y_dict = prepare_corpus(YOGI_URLS)

    lda_p = run_lda(p_corpus, p_dict)
    lda_y = run_lda(y_corpus, y_dict)

    return p_texts, y_texts, lda_p, lda_y

p_texts, y_texts, lda_p, lda_y = load_models()

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

    words = []
    weights = []

    for topic in topics:
        for word, weight in topic[1]:
            words.append(word)
            weights.append(weight)

    fig, ax = plt.subplots()
    ax.barh(words, weights)
    ax.set_title(title)
    st.pyplot(fig)

def venn_diagram(p_texts, y_texts):
    p_words = set([word for text in p_texts for word in text])
    y_words = set([word for text in y_texts for word in text])

    fig, ax = plt.subplots()
    venn2([p_words, y_words], set_labels=("Pinarayi", "Yogi"))
    st.pyplot(fig)

# -----------------------------
# UI CONFIG
# -----------------------------
st.set_page_config(
    page_title="Political Topic Analysis",
    layout="wide"
)

st.title("🧠 Political Speech Topic Analysis Dashboard")

section = st.sidebar.radio("Select View", [
    "Topics",
    "Word Clouds",
    "Topic Importance",
    "Comparison"
])

# -----------------------------
# UI SECTIONS
# -----------------------------
if section == "Topics":
    option = st.selectbox(
        "Select Politician",
        ["Pinarayi Vijayan", "Yogi Adityanath"]
    )

    def show_topics(model):
        topics = model.print_topics(-1)
        for i, topic in topics:
            st.write(f"### Topic {i}")
            st.write(topic)

    if option == "Pinarayi Vijayan":
        show_topics(lda_p)
    else:
        show_topics(lda_y)

elif section == "Word Clouds":
    col1, col2 = st.columns(2)
    with col1:
        generate_wordcloud(p_texts, "Pinarayi Vijayan")
    with col2:
        generate_wordcloud(y_texts, "Yogi Adityanath")

elif section == "Topic Importance":
    col1, col2 = st.columns(2)
    with col1:
        show_topic_bar(lda_p, "Pinarayi Vijayan")
    with col2:
        show_topic_bar(lda_y, "Yogi Adityanath")

elif section == "Comparison":
    st.subheader("Vocabulary Overlap")
    venn_diagram(p_texts, y_texts)
