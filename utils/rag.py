# utils/rag.py
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

RAG_FOLDER = "rag_data"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small & fast

class RAGIndex:
    def __init__(self, chunk_size=800, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
        self.chunks = []          # list of chunk texts
        self.meta = []            # list of metadata dicts (filename, idx)
        self.vectors = None       # numpy array of embeddings

    def _chunk_text(self, text: str):
        chunks = []
        i = 0
        L = len(text)
        while i < L:
            end = i + self.chunk_size
            chunks.append(text[i:end])
            i += self.chunk_size - self.chunk_overlap
        return chunks

    def index_all_files(self):
        """Load all files from rag_data and build embeddings (in-memory)."""
        files = sorted(os.listdir(RAG_FOLDER))
        all_chunks = []
        all_meta = []
        for fname in files:
            path = os.path.join(RAG_FOLDER, fname)
            if os.path.isdir(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    txt = f.read()
            except Exception:
                # try binary decode fallback
                with open(path, "rb") as f:
                    txt = f.read().decode("utf-8", errors="ignore")

            file_chunks = self._chunk_text(txt)
            for i, ch in enumerate(file_chunks):
                all_chunks.append(ch)
                all_meta.append({"source": fname, "chunk_index": i, "length": len(ch)})

        if len(all_chunks) == 0:
            self.chunks = []
            self.meta = []
            self.vectors = None
            return

        # compute embeddings in batch
        embeddings = self.embedder.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)
        self.chunks = all_chunks
        self.meta = all_meta
        self.vectors = embeddings
        print(f"Indexed {len(self.chunks)} chunks from {len(files)} files.")

    def retrieve(self, query: str, top_k: int = 4):
        """Return top_k chunks and their metadata for the query."""
        if self.vectors is None or len(self.chunks) == 0:
            return []

        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        scores = cosine_similarity(q_emb, self.vectors)[0]  # shape (n,)
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_idx:
            results.append({
                "score": float(scores[idx]),
                "text": self.chunks[idx],
                "meta": self.meta[idx]
            })
        return results

# module-level RAG instance (created on import)
rag_index = RAGIndex()
