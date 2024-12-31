# 📰 Article Recommender System 🚀

Welcome to the **AI-Powered Article Recommender System**! This project combines cutting-edge **Natural Language Processing (NLP)** techniques with a sleek web interface to provide personalized article recommendations based on user queries. 

---

## 📚 Features
- 🔍 **Semantic Search**: Recommend articles based on the user's search query using advanced embeddings.
- 🏢 **Filter by Publisher**: Dynamically filter recommendations by article publishers.
- ⚡ **Fast & Accurate**: Powered by the `all-mpnet-base-v2` embedding model for high performance.
- 🎨 **Intuitive UI**: Simple, modern web interface for seamless interaction.

---

## 🚀 Technologies Used
- **Backend**: Python, Flask  
- **NLP Models**: [SentenceTransformers](https://www.sbert.net/)  
- **Database**: PostgreSQL  
- **Frontend**: HTML, CSS, JavaScript  
- **AI Techniques**: Text embeddings, cosine similarity

---

## 🛠️ Project Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/article-recommender.git
cd article-recommender
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Setup the Database
Install and configure PostgreSQL.
Create a database and populate it using the script provided:
```bash
python populate_database.py
```
### 4️⃣ Generate Embeddings
To generate embeddings for your dataset:
```bash
python embed_articles.py
```
Note: The embeddings and database files were not uploaded due to GitHub's file size restrictions. Use the provided scripts to generate them.

### 5️⃣ Run the Application
```bash
flask run
```
Access the app at http://localhost:5000.
