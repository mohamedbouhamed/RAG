# prepare_embeddings.py
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import pickle

# Charger tous les chunks (copier le code de traitement des PDFs)
# ... [ton code de traitement des PDFs] ...

# Créer les embeddings
embedding_model = SentenceTransformerEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")
all_chunks = [chunk['content'] for chunk in chunks]

# Sauvegarder
db = Chroma.from_texts(all_chunks, embedding=embedding_model, persist_directory="./chroma_db")
print("✅ Embeddings saved to ./chroma_db")