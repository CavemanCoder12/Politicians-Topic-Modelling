import nltk
import re
import streamlit as st
import subprocess
import sys
import spacy
from newspaper import Article
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import feedparser
from urllib.parse import quote

nltk.download('stopwords', quiet=True)

try:
    nlp = spacy.load("en_core_web_sm")
except:
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

stop_words = set(stopwords.words('english'))

def extract_article(url):
    try:
        a = Article(url)
        a.download()
        a.parse()
        return a.text
    except:
        return ""

def preprocess(text):
    text = re.sub(r'\W', ' ', text.lower())
    doc = nlp(text)
    return [t.lemma_ for t in doc if t.text not in stop_words and len(t.text) > 3]

def prepare_corpus(urls):
    texts = []
    for u in urls:
        t = extract_article(u)
        tokens = preprocess(t)
        if tokens:
            texts.append(tokens)

    if not texts:
        return [], [], None

    d = corpora.Dictionary(texts)
    c = [d.doc2bow(x) for x in texts]
    return texts, c, d

def run_lda(corpus, dictionary):
    if not corpus or dictionary is None:
        return None
    return LdaModel(corpus=corpus, id2word=dictionary, num_topics=4, passes=10)

def load_models_dynamic(urls1, urls2):
    p_texts, p_corpus, p_dict = prepare_corpus(urls1)
    y_texts, y_corpus, y_dict = prepare_corpus(urls2)
    return p_texts, y_texts, run_lda(p_corpus, p_dict), run_lda(y_corpus, y_dict)

def fetch_news_urls(query):
    query = quote(query)
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    return [e.link for e in feed.entries[:5]]

st.title("Topic Modelling Dashboard")

q1 = st.text_input("Politician 1", "Pinarayi Vijayan")
q2 = st.text_input("Politician 2", "Yogi Adityanath")

if st.button("Fetch & Analyze"):
    urls1 = fetch_news_urls(q1)
    urls2 = fetch_news_urls(q2)
    st.session_state["data"] = load_models_dynamic(urls1, urls2)

if "data" not in st.session_state:
    st.stop()

p_texts, y_texts, lda_p, lda_y = st.session_state["data"]

def show_topics(model):
    if model:
        for i, t in model.print_topics():
            st.write(t)

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
