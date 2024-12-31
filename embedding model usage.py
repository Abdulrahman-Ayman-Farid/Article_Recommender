# Used to embed the articles using pre-trained model
import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer

# Database configuration (put your confgs)
db_config = {
    "dbname": "articles_db",
    "user": "postgres",
    "password": "1234",
    "host": "localhost",
    "port": "5432"
}

# Define the model
model = SentenceTransformer('all-mpnet-base-v2', device='cuda') # High-quality embedding model using GPU , if you don't have cuda installed , remove (,device ='cuda')

# Connect to the PostgreSQL database
conn = psycopg2.connect(**db_config)
cursor = conn.cursor()

# Fetch articles from the database
query = "SELECT id, title, publisher, link FROM articles"  # Adjust based on your schema
cursor.execute(query)
articles = cursor.fetchall()

# Separate metadata and titles
article_ids = [row[0] for row in articles]
article_titles = [row[1] for row in articles]
article_publishers = [row[2] for row in articles]
article_links = [row[3] for row in articles]

# Generate embeddings for article titles
print("Generating embeddings using all-mpnet-base-v2. This may take a while...")
embeddings = model.encode(article_titles, show_progress_bar=True, batch_size=16)  # Optimized for performance

# Save embeddings and metadata
np.save('article_embeddings.npy', embeddings)
metadata = [{"id": id_, "title": title, "publisher": publisher, "link": link} for id_, title, publisher, link in zip(article_ids, article_titles, article_publishers, article_links)]
np.save('article_metadata.npy', metadata)

print("Embeddings and metadata saved successfully!")

# Close the database connection
cursor.close()
conn.close()
