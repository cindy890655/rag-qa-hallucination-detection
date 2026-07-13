from sentence_transformers import SentenceTransformer

print("Loading model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded successfully!")

print(f"Embedding dimension: {model.get_embedding_dimension()}")