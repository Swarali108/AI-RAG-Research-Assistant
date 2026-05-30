from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.pdf_loader import load_pdf
from src.chunking import recursive_chunking
from src.embedding import EmbeddingModel
from src.vector_store import VectorStore


pdf_path = project_root / "docs" / "sample.pdf"

pages = load_pdf(pdf_path)
chunks = recursive_chunking(pages)

texts = [chunk["text"] for chunk in chunks]

embedding_model = EmbeddingModel()

chunk_embeddings = embedding_model.embed_texts(texts)

dimension = chunk_embeddings.shape[1]
vector_store = VectorStore(dimension)
vector_store.add_embeddings(chunk_embeddings, chunks)

query = "What is artificial intelligence?"
query_embedding = embedding_model.embed_query(query)

results = vector_store.search(query_embedding, top_k=3)

print(f"Query: {query}")
print(f"Results found: {len(results)}")

for result in results:
    print("\nScore:", result["score"])
    print("Page:", result["page"])
    print("Chunk ID:", result["chunk_id"])
    print("Text:", result["text"][:500])
    