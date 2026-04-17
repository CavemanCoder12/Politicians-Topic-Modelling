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

# Dynamic news
import feedparser
from urllib.parse import quote

# -----------------------------
# SETUP
# -----------------------------
nltk.download('stopwords', quiet=True)

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
# CORE FUNCTIONS
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
    text = re.sub(r'\W', ' ', text.lower())
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
        if tokens:
            texts.append(tokens)

    if not texts:
        return [], [], None

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    return texts, corpus, dictionary

def run_lda(corpus, dictionary):
    if not corpus or dictionary is None:
        return None
    return LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=4,
        passes=10,
        random_state=42
    )

def load_models_dynamic(urls1, urls2):
    p_texts, p_corpus, p_dict = prepare_corpus(urls1)
    y_texts, y_corpus, y_dict = prepare_corpus(urls2)

    lda_p = run_lda(p_corpus, p_dict)
    lda_y = run_lda(y_corpus, y_dict)

    return p_texts, y_texts, lda_p, lda_y

def fetch_news_urls(query, num_articles=5):
    try:
        query = quote(query)
        url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(url)
        return [entry.link for entry in feed.entries[:num_articles]]
    except:
        return []

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Topic Modelling Dashboard", layout="wide")
st.title("🧠 Political Speech Topic Analysis")

query1 = st.text_input("Enter Politician 1", "Pinarayi Vijayan")
query2 = st.text_input("Enter Politician 2", "Yogi Adityanath")

if st.button("Fetch & Analyze", key="fetch_main"):
    with st.spinner("Fetching and processing..."):
        urls1 = fetch_news_urls(query1)
        urls2 = fetch_news_urls(query2)

        if not urls1 or not urls2:
            st.error("Could not fetch articles. Try different keywords.")
        else:
            p_texts, y_texts, lda_p, lda_y = load_models_dynamic(urls1, urls2)

            if lda_p is None or lda_y is None:
                st.error("Not enough usable content for topic modelling.")
            else:
                st.session_state["data"] = (p_texts, y_texts, lda_p, lda_y)

# Stop until data is ready
if "data" not in st.session_state:
    st.info("Enter names and click 'Fetch & Analyze'")
    st.stop()

p_texts, y_texts, lda_p, lda_y = st.session_state["data"]

# -----------------------------
# VISUAL FUNCTIONS
# -----------------------------
def show_topics(model):
    if model is None:
        st.warning("Not enough data to generate topics")
        return
    for i, topic in model.print_topics(-1):
        st.write(f"### Topic {i}")
        st.write(topic)

def generate_wordcloud(texts, title):
    if not texts:
        st.warning(f"No data available for {title}")
        return
    all_words = " ".join([" ".join(text) for text in texts])
    wc = WordCloud(width=800, height=400).generate(all_words)

    fig, ax = plt.subplots()
    ax.imshow(wc)
    ax.axis("off")
    ax.set_title(title)
    st.pyplot(fig)

def venn_diagram(p_texts, y_texts):
    if not p_texts or not y_texts:
        st.warning("Not enough data for comparison")
        return
    p_words = set(word for text in p_texts for word in text)
    y_words = set(word for text in y_texts for word in text)

    fig, ax = plt.subplots()
    venn2([p_words, y_words], set_labels=("Politician 1", "Politician 2"))
    st.pyplot(fig)

# -----------------------------
# NAVIGATION
# -----------------------------
section = st.sidebar.radio("Select View", [
    "Topics",
    "Word Clouds",
    "Comparison"
])

if section == "Topics":
    choice = st.selectbox("Select", ["Politician 1", "Politician 2"])
    show_topics(lda_p if choice == "Politician 1" else lda_y)

elif section == "Word Clouds":
    col1, col2 = st.columns(2)
    with col1:
        generate_wordcloud(p_texts, "Politician 1")
    with col2:
        generate_wordcloud(y_texts, "Politician 2")

elif section == "Comparison":
    venn_diagram(p_texts, y_texts)
