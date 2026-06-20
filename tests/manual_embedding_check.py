from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.pdf_loader import load_pdf
from src.chunking import recursive_chunking
from src.embedding import EmbeddingModel


pdf_path = project_root / "docs" / "sample.pdf"

pages = load_pdf(pdf_path)
chunks = recursive_chunking(pages)

texts = [chunk["text"] for chunk in chunks[:5]]

embedding_model = EmbeddingModel()
embeddings = embedding_model.embed_texts(texts)

print(f"Total test chunks: {len(texts)}")
print(f"Embedding shape: {embeddings.shape}")
print(f"First embedding first 5 values: {embeddings[0][:5]}")
