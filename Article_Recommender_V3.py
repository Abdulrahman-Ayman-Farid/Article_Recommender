import os
import re
import numpy as np
import pandas as pd
import torch
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
from sqlalchemy import create_engine
from sqlalchemy.sql import text
from flask import Flask, render_template, request

class ArticleRecommender:
    def __init__(self, db_config, embeddings_file=r"E:\Programming\ArticleRecommender\Articles Recommender\embeddings file\article_embeddings.npy"): # put the path to article_embeddings.npy
        self.db_config = db_config
        self.embeddings_file = embeddings_file
        self.articles = self._fetch_articles_from_db()
        self.model = SentenceTransformer('all-mpnet-base-v2', device='cuda')  # model
        self.embeddings = None

        # Check if embeddings need to be computed
        if not self._embeddings_exist():
            print("Embeddings not found or dataset changed, computing embeddings...")
            self.embeddings = self._embed_articles()
            self._save_embeddings()
        else:
            print("Embeddings found. Loading from disk...")
            self.embeddings = np.load(self.embeddings_file)

    def _embeddings_exist(self):
        if os.path.exists(self.embeddings_file):
            saved_embeddings = np.load(self.embeddings_file)
            saved_article_count = saved_embeddings.shape[0]
            current_article_count = len(self.articles)
            if saved_article_count == current_article_count:
                return True
        return False

    def _connect_to_db(self):
        db_url = f"postgresql://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['dbname']}"
        return create_engine(db_url)

    def _fetch_articles_from_db(self):
        print("Fetching articles from the database...")
        engine = self._connect_to_db()
        with engine.connect() as connection:
            query = text("SELECT id, title, publisher, link FROM articles;")
            articles = pd.read_sql(query, connection)
        return articles.to_dict('records')

    def preprocess_text(self, text):
        if not text:
            return ""
        stop_words = set(stopwords.words('english'))
        text = re.sub(r'\W', ' ', text)
        text = ' '.join(word for word in text.split() if word.lower() not in stop_words)
        return text.lower()

    def _embed_articles(self):
        print("Embedding articles...")
        batch_size = 128
        embeddings = []
        contents = [self.preprocess_text(article['title'] + ' ' + article.get('publisher', '')) for article in self.articles]

        for i in range(0, len(contents), batch_size):
            batch = contents[i:i + batch_size]
            print(f"Processing batch {i // batch_size + 1}/{len(contents) // batch_size + 1}...")
            batch_embeddings = self.model.encode(batch, convert_to_tensor=True, show_progress_bar=True)
            embeddings.append(batch_embeddings.cpu().numpy())

        embeddings = np.concatenate(embeddings, axis=0)
        print(f"Embedding completed. Shape: {embeddings.shape}")
        return embeddings

    def _save_embeddings(self):
        print(f"Saving embeddings to {self.embeddings_file}...")
        np.save(self.embeddings_file, self.embeddings)
        print("Embeddings saved.")

    def recommend(self, query, top_n=15, threshold=0.5):
        if self.embeddings is None or len(self.embeddings) == 0:
            print("No articles available for recommendations.")
            return []

        query = self.preprocess_text(query)
        query_embedding = self.model.encode(query, convert_to_tensor=True, device='cuda')

        similarities = util.cos_sim(query_embedding, torch.tensor(self.embeddings, device='cuda')).cpu().numpy().flatten()

        for i, article in enumerate(self.articles):
            article["score"] = similarities[i]

        filtered_articles = [article for article in self.articles if article["score"] > threshold]
        sorted_articles = sorted(filtered_articles, key=lambda x: x["score"], reverse=True)

        return sorted_articles[:top_n]

app = Flask(__name__)

class ArticleRecommenderApp:
    def __init__(self, app, db_config):
        self.app = app
        self.db_config = db_config
        self.recommender = ArticleRecommender(db_config)

    def get_recommendations(self, query):
        return self.recommender.recommend(query)

db_config = {
    "dbname": "articles_db",
    "user": "postgres",
    "password": "1234",
    "host": "localhost",
    "port": "5432"
}

recommender_app = ArticleRecommenderApp(app, db_config)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        query = request.form.get("query")
        selected_publisher = request.form.get("publisher")
        if query:
            recommendations = recommender_app.get_recommendations(query)
            if selected_publisher:
                recommendations = [
                    article for article in recommendations
                    if article['publisher'] == selected_publisher
                ]
            publishers = list(set([article['publisher'] for article in recommendations if 'publisher' in article]))
            return render_template(
                "index.html",
                query=query,
                recommendations=recommendations,
                publishers=publishers,
                selected_publisher=selected_publisher
            )
    return render_template("index.html", query=None, recommendations=None, publishers=[])

if __name__ == "__main__":
    app.run(debug=True)
