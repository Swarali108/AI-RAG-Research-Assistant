from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.pdf_loader import load_pdf
from src.chunking import recursive_chunking
from src.embedding import EmbeddingModel
from src.vector_store import VectorStore
from src.rag import RAGPipeline


pdf_path = project_root / "docs" / "sample.pdf"

pages = load_pdf(pdf_path)
chunks = recursive_chunking(pages)

texts = [chunk["text"] for chunk in chunks]

embedding_model = EmbeddingModel()
chunk_embeddings = embedding_model.embed_texts(texts)

vector_store = VectorStore(chunk_embeddings.shape[1])
vector_store.add_embeddings(chunk_embeddings, chunks)

rag = RAGPipeline(embedding_model, vector_store)

question = "What is artificial intelligence?"
result = rag.ask(question)

print("Question:", question)
print("\nAnswer:")
print(result["answer"])

print("\nCitations:")
for citation in result["citations"]:
    print(citation)
    