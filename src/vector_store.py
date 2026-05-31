import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks = []

    def add_embeddings(self, embeddings, chunks):
        embeddings = np.array(embeddings).astype("float32")
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding, top_k=3):
        query_embedding = np.array(query_embedding).astype("float32")
        scores, indices = self.index.search(query_embedding, top_k)

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue

            chunk = self.chunks[index]

            results.append({
                "score": float(score),
                "source": chunk.get("source", "uploaded_document.pdf"),
                "chunk_id": chunk["chunk_id"],
                "page": chunk["page"],
                "text": chunk["text"]
            })

        return results
    