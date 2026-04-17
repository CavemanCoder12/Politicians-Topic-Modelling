import nltk
import re
import streamlit as st
import subprocess
import sys
import spacy
import requests

# NLP
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords

# Visuals
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

# -----------------------------
# SETUP
# -----------------------------
nltk.download('stopwords', quiet=True)

try:
    nlp = spacy.load("en_core_web_sm")
except:
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

stop_words = set(stopwords.words('english'))

# -----------------------------
# NEWS API FETCH
# -----------------------------
def fetch_news_articles(query, num_articles=5):
    API_KEY = st.secrets["NEWS_API_KEY"]

    url = "https://newsapi.org/v2/everything"

    params = {
        "q": query,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": num_articles,
        "apiKey": API_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()

    articles = []

    if data["status"] == "ok":
        for item in data["articles"]:
            text = (item.get("title", "") + " " +
                    item.get("description", "") + " " +
                    item.get("content", ""))
            if len(text) > 100:
                articles.append(text)

    return articles

# -----------------------------
# NLP PIPELINE
# -----------------------------
def preprocess(text):
    text = re.sub(r'\W', ' ', text.lower())
    doc = nlp(text)
    return [
        token.lemma_
        for token in doc
        if token.text not in stop_words and len(token.text) > 3
    ]

def prepare_corpus(texts):
    processed = [preprocess(t) for t in texts if t]

    if not processed:
        return [], [], None

    dictionary = corpora.Dictionary(processed)
    corpus = [dictionary.doc2bow(text) for text in processed]

    return processed, corpus, dictionary

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

st.subheader("Key Insight")

st.write("Politician 1 focuses more on:", p_texts[0][:5])
st.write("Politician 2 focuses more on:", y_texts[0][:5])

st.write(f"Articles fetched: {len(p_texts)} vs {len(y_texts)}")

def load_models_dynamic(texts1, texts2):
    p_texts, p_corpus, p_dict = prepare_corpus(texts1)
    y_texts, y_corpus, y_dict = prepare_corpus(texts2)

    lda_p = run_lda(p_corpus, p_dict)
    lda_y = run_lda(y_corpus, y_dict)

    return p_texts, y_texts, lda_p, lda_y

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Topic Modelling Dashboard", layout="wide")
st.title("🧠 Political Topic Analysis (News API Powered)")

q1 = st.text_input("Enter Politician 1", "Pinarayi Vijayan")
q2 = st.text_input("Enter Politician 2", "Yogi Adityanath")

if st.button("Fetch & Analyze"):
    with st.spinner("Fetching news from API..."):
        texts1 = fetch_news_articles(q1)
        texts2 = fetch_news_articles(q2)

        if not texts1 or not texts2:
            st.error("No articles found. Try different keywords.")
        else:
            p_texts, y_texts, lda_p, lda_y = load_models_dynamic(texts1, texts2)
            st.session_state["data"] = (p_texts, y_texts, lda_p, lda_y)

if "data" not in st.session_state:
    st.stop()

p_texts, y_texts, lda_p, lda_y = st.session_state["data"]

# -----------------------------
# VISUALS
# -----------------------------
def show_topics(model):
    if model is None:
        st.warning("Not enough data")
        return
    for i, topic in model.print_topics():
        st.write(topic)

def wordcloud(texts):
    if not texts:
        return
    wc = WordCloud().generate(" ".join([" ".join(t) for t in texts]))
    plt.imshow(wc)
    plt.axis("off")
    st.pyplot(plt)

def venn(p, y):
    if not p or not y:
        return
    fig, ax = plt.subplots()
    venn2([set(sum(p, [])), set(sum(y, []))], ("P1", "P2"))
    st.pyplot(fig)

view = st.sidebar.selectbox("View", ["Topics", "Wordcloud", "Compare"])

if view == "Topics":
    show_topics(lda_p)
    show_topics(lda_y)

elif view == "Wordcloud":
    wordcloud(p_texts)
    wordcloud(y_texts)

else:
    venn(p_texts, y_texts)
