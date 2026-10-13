# 🧠 Backend Documentation

This document thoroughly covers the backend of the **Article Recommender** application, including its architecture, database configuration, embedding pipeline, core modules, API routes, and dependencies.

---

## 📖 1. Overview

The Article Recommender is a **Flask-based article recommendation system** that uses **PostgreSQL** and **sentence-transformer embeddings** to deliver semantic search over a corpus of articles.

At its core, the application converts article titles into dense vector embeddings using the [`all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) model — a high-quality sentence embedding model from the [SentenceTransformers](https://www.sbert.net/) framework. When a user submits a search query, the query is embedded using the same model and compared against the stored article embeddings using **cosine similarity**. The most semantically similar articles are returned as recommendations.

This approach enables **semantic search**: articles are matched by meaning rather than by exact keyword overlap, so a query like *"machine learning trends"* can surface articles titled *"AI advancements in 2024"* even when no words are shared.

---

## 🏗️ 2. Architecture

The backend follows a **two-script architecture** that cleanly separates the (expensive, offline) embedding generation step from the (interactive, online) web-serving step.

### `embedding model usage.py`
A **standalone script** that connects to PostgreSQL, fetches article titles, generates embeddings using the `SentenceTransformer('all-mpnet-base-v2')` model, and saves them to two `.npy` files:
- `article_embeddings.npy` — the embedding vectors
- `article_metadata.npy` — a list of metadata dicts (`id`, `title`, `publisher`, `link`)

This script is intended to be run once (or whenever the article corpus changes) to pre-compute the embeddings that the web app will consume.

### `Article_Recommender_V3.py`
The **main Flask application**. It loads the pre-computed embeddings (computing them on the fly if they are missing or stale), connects to the database to fetch live article metadata, and serves recommendations through web routes rendered with Jinja2 templates.

### Architecture Diagram

```
┌─────────────────┐
│  PostgreSQL DB  │
│  (articles_db)  │
│  articles table │
└────────┬────────┘
         │ SELECT id, title, publisher, link
         ▼
┌─────────────────────────────┐
│  embedding model usage.py   │
│  SentenceTransformer        │
│  ('all-mpnet-base-v2')      │
└────────┬────────────────────┘
         │ np.save(...)
         ▼
┌─────────────────────────────────────────┐
│  article_embeddings.npy                  │
│  article_metadata.npy                    │
└────────┬────────────────────────────────┘
         │ np.load(...)
         ▼
┌─────────────────────────────┐
│  Article_Recommender_V3.py  │
│  (Flask application)        │
│  ArticleRecommender class   │
│  ArticleRecommenderApp class│
└────────┬────────────────────┘
         │ render_template(...)
         ▼
┌──────────────────────────────┐
│  Web UI                      │
│  templates/index.html        │
│  templates/read.html         │
└──────────────────────────────┘
```

---

## 🗄️ 3. Database Configuration

Both scripts connect to the same PostgreSQL database using a shared `db_config` dictionary. It contains the keys `dbname`, `user`, `password`, `host`, and `port`:

```python
db_config = {
    "dbname": "articles_db",
    "user": "postgres",
    "password": "1234",
    "host": "localhost",
    "port": "5432"
}
```

The database name is **`articles_db`**, and the application relies on an **`articles`** table with the following columns:

| Column       | Description                                                        |
|--------------|--------------------------------------------------------------------|
| `id`         | Unique identifier for the article (primary key).                   |
| `title`      | The article title — the primary text used to generate embeddings.  |
| `publisher`  | The publisher/source of the article; used for filtering results.   |
| `link`       | The URL pointing to the original article.                          |

> ⚠️ **Security Note:** The configuration above hardcodes the password (`"1234"`) for simplicity. In a **production** environment, the password (and other credentials) should **never** be committed to source. Replace them with **environment variables** or a **secrets management** solution, for example:
>
> ```python
> import os
> db_config = {
>     "dbname": os.environ["DB_NAME"],
>     "user": os.environ["DB_USER"],
>     "password": os.environ["DB_PASSWORD"],
>     "host": os.environ["DB_HOST"],
>     "port": os.environ.get("DB_PORT", "5432")
> }
> ```

---

## ⚙️ 4. Embedding Pipeline

The `embedding model usage.py` script is responsible for pre-computing and persisting article embeddings. Its workflow is as follows:

### Step 1 — Connect to the database
Connects to PostgreSQL using `psycopg2` via `psycopg2.connect(**db_config)`:

```python
conn = psycopg2.connect(**db_config)
cursor = conn.cursor()
```

### Step 2 — Fetch articles
Executes the SQL query to retrieve article rows:

```python
query = "SELECT id, title, publisher, link FROM articles"
cursor.execute(query)
articles = cursor.fetchall()
```

### Step 3 — Separate metadata and titles
Splits the fetched rows into parallel lists of `article_ids`, `article_titles`, `article_publishers`, and `article_links`:

```python
article_ids = [row[0] for row in articles]
article_titles = [row[1] for row in articles]
article_publishers = [row[2] for row in articles]
article_links = [row[3] for row in articles]
```

### Step 4 — Generate embeddings
Uses the `all-mpnet-base-v2` model to generate vector embeddings for the article titles, with a progress bar and a batch size of 16:

```python
embeddings = model.encode(article_titles, show_progress_bar=True, batch_size=16)
```

### Step 5 — Save outputs as `.npy` files
Saves the embedding vectors and a list of metadata dicts to disk:

```python
np.save('article_embeddings.npy', embeddings)
metadata = [{"id": id_, "title": title, "publisher": publisher, "link": link}
            for id_, title, publisher, link in
            zip(article_ids, article_titles, article_publishers, article_links)]
np.save('article_metadata.npy', metadata)
```

- **`article_embeddings.npy`** — the embedding vectors (a 2D NumPy array of shape `(num_articles, embedding_dim)`).
- **`article_metadata.npy`** — a list of metadata dicts, each containing `id`, `title`, `publisher`, and `link`.

### Step 6 — GPU acceleration
The model is instantiated with GPU acceleration via `device='cuda'`:

```python
model = SentenceTransformer('all-mpnet-base-v2', device='cuda')
```

> 💡 **No CUDA GPU?** If you do not have a CUDA-compatible GPU available, simply **remove the `device='cuda'` argument** so the model falls back to CPU:
>
> ```python
> model = SentenceTransformer('all-mpnet-base-v2')
> ```

Finally, the script closes the database connection:

```python
cursor.close()
conn.close()
```

---

## 🧩 5. Core Backend Modules

The `Article_Recommender_V3.py` file defines two classes: `ArticleRecommender` (the recommendation engine) and `ArticleRecommenderApp` (the Flask web layer that wraps it).

### `ArticleRecommender` Class

The core recommendation engine. It manages the database connection, the embedding model, the cached embeddings, and the similarity computation.

#### `__init__(self, db_config, embeddings_file=r"E:\Programming\ArticleRecommender\Articles Recommender\embeddings file\article_embeddings.npy")`

Initializes the recommender.

- **Parameters:**
  - `db_config` — dictionary of PostgreSQL connection parameters.
  - `embeddings_file` — path to the `article_embeddings.npy` file. *(The default is a Windows absolute path — update it to point to your local embeddings file.)*
- **Behavior:**
  - Stores the config and embeddings path.
  - Fetches articles from the database via `_fetch_articles_from_db()`.
  - Loads the `SentenceTransformer('all-mpnet-base-v2')` model (on `cuda`).
  - If embeddings do not exist (or are stale), computes them via `_embed_articles()` and saves them with `_save_embeddings()`; otherwise loads them from disk with `np.load`.

```python
def __init__(self, db_config, embeddings_file=r"E:\Programming\ArticleRecommender\Articles Recommender\embeddings file\article_embeddings.npy"):
    self.db_config = db_config
    self.embeddings_file = embeddings_file
    self.articles = self._fetch_articles_from_db()
    self.model = SentenceTransformer('all-mpnet-base-v2', device='cuda')
    self.embeddings = None

    if not self._embeddings_exist():
        print("Embeddings not found or dataset changed, computing embeddings...")
        self.embeddings = self._embed_articles()
        self._save_embeddings()
    else:
        print("Embeddings found. Loading from disk...")
        self.embeddings = np.load(self.embeddings_file)
```

#### `_embeddings_exist(self)`

Checks whether the embeddings file is present **and** whether the saved embedding count matches the current article count.

- **Returns:** `True` if the file exists and `saved_embeddings.shape[0] == len(self.articles)`; otherwise `False`.

```python
def _embeddings_exist(self):
    if os.path.exists(self.embeddings_file):
        saved_embeddings = np.load(self.embeddings_file)
        saved_article_count = saved_embeddings.shape[0]
        current_article_count = len(self.articles)
        if saved_article_count == current_article_count:
            return True
    return False
```

#### `_connect_to_db(self)`

Establishes a PostgreSQL connection using SQLAlchemy's `create_engine`.

- **Returns:** A SQLAlchemy `Engine` constructed from the `db_config` credentials.

```python
def _connect_to_db(self):
    db_url = f"postgresql://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['dbname']}"
    return create_engine(db_url)
```

#### `_fetch_articles_from_db(self)`

Retrieves article metadata via `SELECT id, title, publisher, link FROM articles` and returns the rows as a list of dicts.

- **Returns:** A list of dictionaries (one per article), produced by `pandas.read_sql(...).to_dict('records')`.

```python
def _fetch_articles_from_db(self):
    print("Fetching articles from the database...")
    engine = self._connect_to_db()
    with engine.connect() as connection:
        query = text("SELECT id, title, publisher, link FROM articles;")
        articles = pd.read_sql(query, connection)
    return articles.to_dict('records')
```

#### `preprocess_text(self, text)`

Cleans text by removing non-word characters, removing English stopwords (via NLTK), and lowercasing.

- **Parameters:**
  - `text` — the raw input string.
- **Returns:** The cleaned, lowercased string with stopwords removed (or `""` if the input is falsy).

```python
def preprocess_text(self, text):
    if not text:
        return ""
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'\W', ' ', text)
    text = ' '.join(word for word in text.split() if word.lower() not in stop_words)
    return text.lower()
```

#### `_embed_articles(self)`

Generates embeddings for the fetched articles in **batches of 128**, preprocessing each article's `title + publisher` before encoding.

- **Returns:** A concatenated NumPy array of embeddings (moved off the GPU via `.cpu().numpy()`).

```python
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
```

#### `_save_embeddings(self)`

Saves the newly computed embeddings to disk via `np.save`.

```python
def _save_embeddings(self):
    print(f"Saving embeddings to {self.embeddings_file}...")
    np.save(self.embeddings_file, self.embeddings)
    print("Embeddings saved.")
```

#### `recommend(self, query, top_n=15, threshold=0.5)`

Computes cosine similarity between the query embedding and the article embeddings, filters by a similarity threshold, sorts descending, and returns the top matches.

- **Parameters:**
  - `query` — the user's search string.
  - `top_n` — maximum number of recommendations to return (default `15`).
  - `threshold` — minimum cosine similarity score for an article to be included (default `0.5`).
- **Returns:** A list of article dicts (each annotated with a `"score"` key), sorted by score descending, truncated to `top_n`. Returns `[]` if no embeddings are available.

```python
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
```

### `ArticleRecommenderApp` Class

The Flask web layer that wraps the `ArticleRecommender` engine and exposes it through view functions.

#### `__init__(self, app, db_config)`

Initializes the Flask app and creates an `ArticleRecommender` instance.

- **Parameters:**
  - `app` — the Flask application instance.
  - `db_config` — dictionary of PostgreSQL connection parameters.
- **Behavior:** Stores the app and config, then instantiates `self.recommender = ArticleRecommender(db_config)`.

```python
def __init__(self, app, db_config):
    self.app = app
    self.db_config = db_config
    self.recommender = ArticleRecommender(db_config)
```

#### `get_recommendations(self, query)`

Interfaces with the `ArticleRecommender.recommend()` method to fetch results.

- **Parameters:**
  - `query` — the user's search string.
- **Returns:** The list of recommended articles returned by `self.recommender.recommend(query)`.

```python
def get_recommendations(self, query):
    return self.recommender.recommend(query)
```

#### `index()`

The Flask view function handling `GET` and `POST` requests for the main route (`/`).

- **On POST:**
  - Extracts `query` and `publisher` from the submitted form data.
  - If a `query` is present, fetches recommendations via `get_recommendations(query)`.
  - If a `publisher` was selected, filters the recommendations to only those whose `publisher` matches.
  - Builds a list of unique `publishers` from the (filtered) recommendations.
  - Renders `templates/index.html` with `query`, `recommendations`, `publishers`, and `selected_publisher`.
- **On GET (or POST without a query):**
  - Renders the empty form with `query=None`, `recommendations=None`, and `publishers=[]`.

```python
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
```

---

## 🌐 6. API Endpoints & Routes

The Flask application exposes the following routes:

| Method | Path   | Description                                                                                          | Template               |
|--------|--------|------------------------------------------------------------------------------------------------------|------------------------|
| GET    | `/`    | Renders the main search form and populates the publisher dropdown.                                   | `templates/index.html` |
| POST   | `/`    | Accepts `query` and `publisher` from form data, fetches recommendations, re-renders `index.html`.    | `templates/index.html` |
| GET    | `/read`| Renders the reading view showing a `selected_article` and a list of `similar_articles`.              | `templates/read.html`  |

### `GET /`

Renders the main search form with an empty state.

- **Form parameters accepted:** None (initial page load).
- **Template context variables:**
  - `query` — `None`
  - `recommendations` — `None`
  - `publishers` — `[]` (empty list)

### `POST /`

Accepts the search form submission, fetches recommendations, optionally filters by publisher, and re-renders the page with results.

- **Form parameters accepted:**
  - `query` — the user's search string (required to trigger recommendations).
  - `publisher` — the selected publisher to filter by (optional).
- **Template context variables (when a query is provided):**
  - `query` — the submitted search string.
  - `recommendations` — the list of matching article dicts (each with a `score`), optionally filtered by publisher.
  - `publishers` — a list of unique publishers derived from the recommendations.
  - `selected_publisher` — the publisher chosen in the form (may be `None`).
- **Template context variables (when no query is provided):**
  - `query` — `None`
  - `recommendations` — `None`
  - `publishers` — `[]`

### `GET /read`

Renders the article reading view.

- **Template context variables:**
  - `selected_article` — the article currently being read.
  - `similar_articles` — a list of articles similar to the selected one.

> ℹ️ **Note:** The `/read` route and its `templates/read.html` template are part of the application's web UI. The `index()` view function in `Article_Recommender_V3.py` handles the `/` route; the `/read` route is served from the templates layer.

---

## 📦 7. Dependencies

The backend relies on the following Python packages:

| Package                 | Description                                                              |
|-------------------------|--------------------------------------------------------------------------|
| `flask`                 | Web framework for serving the application.                               |
| `psycopg2`              | PostgreSQL database adapter (used in the embedding script).             |
| `numpy`                 | Array operations for storing/loading embeddings.                         |
| `sentence-transformers` | Embedding model framework (`all-mpnet-base-v2`).                         |
| `torch`                 | Tensor computation backend for the model.                                |
| `pandas`                | Used in `Article_Recommender_V3.py` for reading SQL into DataFrames.     |
| `sqlalchemy`            | Used for the DB connection (`create_engine`) in the Flask app.           |
| `nltk`                  | Used for English stopwords in text preprocessing.                        |

### Installation

Install all required packages with pip:

```bash
pip install flask psycopg2 numpy sentence-transformers torch pandas sqlalchemy nltk
```

### Hardware Acceleration

The model is instantiated with `device='cuda'`, which requires a **CUDA-compatible GPU**. If no CUDA GPU is available, remove the `device='cuda'` argument from the `SentenceTransformer(...)` constructor in both scripts to fall back to CPU execution.

### NLTK Stopwords Data

The `preprocess_text` method uses NLTK's English stopwords list, which is **not** bundled with the `nltk` package by default. You must download it separately once:

```python
import nltk
nltk.download('stopwords')
```

This can be run in a Python interpreter as a one-time setup step before running the application.
