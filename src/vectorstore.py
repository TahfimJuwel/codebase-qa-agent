"""
Module: vectorstore.py
Manages local vector database indexing and retrieval using Qdrant and SentenceTransformers.
Stores full code chunks alongside metadata (file path, line ranges, chunk type).
"""

from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from src.config import VECTOR_DB_DIR, EMBEDDING_MODEL_NAME, REPO_PATH
from src.parser import CodeChunk, ASTCodeParser
from src.ingest import collect_repo_files, clone_or_update_repo


class CodebaseVectorStore:
    """
    Manages embedding generation and semantic indexing inside an embedded Qdrant database.
    """

    COLLECTION_NAME = "codebase_chunks"

    def __init__(self, db_path: Path = VECTOR_DB_DIR, model_name: str = EMBEDDING_MODEL_NAME):
        self.db_path = db_path
        print(f"[VectorStore] Loading embedding model '{model_name}' on local device...")
        self.embedder = SentenceTransformer(model_name)
        
        # Initialize embedded Qdrant client (stored locally on disk)
        self.client = QdrantClient(path=str(self.db_path))
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Creates the Qdrant collection if it does not already exist."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.COLLECTION_NAME for c in collections)
        
        if not exists:
            # all-MiniLM-L6-v2 outputs vectors of dimension 384
            vector_size = self.embedder.get_sentence_embedding_dimension()
            print(f"[VectorStore] Creating new collection '{self.COLLECTION_NAME}' (dimension={vector_size}, distance=COSINE)...")
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
        else:
            print(f"[VectorStore] Collection '{self.COLLECTION_NAME}' already exists.")

    def index_chunks(self, chunks: List[CodeChunk], batch_size: int = 64) -> int:
        """
        Computes embeddings in batches and uploads points with metadata to Qdrant.
        """
        if not chunks:
            print("[VectorStore] No chunks provided for indexing.")
            return 0

        print(f"[VectorStore] Indexing {len(chunks)} code chunks into Qdrant...")

        total_indexed = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            
            # Text to embed: include chunk_name + content for maximum semantic coverage
            texts_to_embed = [
                f"{c.file_path} - {c.chunk_name}\n{c.content}" for c in batch
            ]
            
            # Compute embeddings
            embeddings = self.embedder.encode(texts_to_embed, show_progress_bar=False, normalize_embeddings=True)
            
            # Construct Qdrant points with payload
            points = []
            for idx, (chunk, embedding) in enumerate(zip(batch, embeddings)):
                point_id = i + idx
                payload = chunk.to_dict()
                
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload=payload
                    )
                )

            # Upload batch to Qdrant
            self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=points
            )
            total_indexed += len(batch)
            print(f"[VectorStore] Indexed {total_indexed}/{len(chunks)} chunks...")

        print("[VectorStore] Indexing complete.")
        return total_indexed

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs semantic similarity search over indexed code chunks.
        Returns top-k chunks with relevance score, file path, lines, and content.
        """
        query_vector = self.embedder.encode(query, normalize_embeddings=True).tolist()
        
        search_results = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_vector,
            limit=top_k
        ).points

        formatted_results = []
        for res in search_results:
            payload = res.payload or {}
            formatted_results.append({
                "score": round(res.score, 4),
                "file_path": payload.get("file_path"),
                "chunk_name": payload.get("chunk_name"),
                "chunk_type": payload.get("chunk_type"),
                "start_line": payload.get("start_line"),
                "end_line": payload.get("end_line"),
                "content": payload.get("content")
            })

        return formatted_results


def build_index():
    """Helper script to clone, parse, and populate the vector store."""
    repo_dir = clone_or_update_repo()
    files = collect_repo_files(repo_dir)
    parser = ASTCodeParser(repo_root=repo_dir)

    all_chunks: List[CodeChunk] = []
    for f in files:
        all_chunks.extend(parser.parse_file(f))

    store = CodebaseVectorStore()
    store.index_chunks(all_chunks)
    return store


if __name__ == "__main__":
    # Test indexing and retrieval
    print("--- Step 1: Building / Loading Index ---")
    store = build_index()

    print("\n--- Step 2: Testing Semantic Search ---")
    test_query = "How does requests handle HTTP basic authentication?"
    results = store.search(test_query, top_k=3)

    for i, r in enumerate(results, start=1):
        print(f"\n[Match {i}] Score: {r['score']}")
        print(f"File: {r['file_path']} (Lines {r['start_line']}-{r['end_line']})")
        print(f"Chunk: {r['chunk_name']} ({r['chunk_type']})")
        print("Code snippet:")
        print("\n".join(r['content'].splitlines()[:6]))