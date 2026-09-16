"""
Phase 2: TF-IDF Lexical Retrieval Knowledge Base for @AmazonHelp.
Indexes 55,011 historical customer-brand resolution pairs with conversation IDs.
Provides high-speed lexical (TF-IDF + Cosine Similarity) retrieval.
"""
import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.config import PROCESSED_DATA_DIR, ARTIFACTS_DIR
from src.taxonomy import mask_pii

logger = logging.getLogger(__name__)

INDEX_CACHE_PATH = ARTIFACTS_DIR / "tfidf_retriever_index.pkl"

class HistoricalRetriever:
    def __init__(
        self, 
        kb_path: Optional[Path] = None, 
        max_features: int = 50000,
        force_rebuild: bool = False
    ):
        self.kb_path = kb_path or (PROCESSED_DATA_DIR / "kb_corpus.parquet")
        self.max_features = max_features
        self.force_rebuild = force_rebuild
        self.df_kb = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self.kb_records: Optional[List[Dict[str, Any]]] = None
        self._load_or_build_index()

    def _load_or_build_index(self) -> None:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        if not self.force_rebuild and INDEX_CACHE_PATH.exists():
            try:
                logger.info("Loading cached retrieval index from %s...", INDEX_CACHE_PATH)
                with open(INDEX_CACHE_PATH, "rb") as f:
                    data = pickle.load(f)
                    self.df_kb = data["df_kb"]
                    self.vectorizer = data["vectorizer"]
                    self.tfidf_matrix = data["tfidf_matrix"]
                    self.kb_records = data.get("kb_records")
                if self.kb_records is None and self.df_kb is not None:
                    self.kb_records = self.df_kb[["conversation_id", "customer_text", "support_reply"]].to_dict("records")
                logger.info("Loaded index with %s historical cases.", f"{len(self.df_kb):,}")
                return
            except Exception as exc:
                logger.warning(
                    "Retriever cache incompatible or corrupted (%s). Self-healing: rebuilding index from %s...",
                    exc, self.kb_path
                )

        logger.info("Building TF-IDF index from %s...", self.kb_path)
        self.df_kb = pd.read_parquet(self.kb_path)
        logger.info("Loaded %s pairs. Vectorizing customer queries...", f"{len(self.df_kb):,}")
        
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=self.max_features,
            stop_words="english",
            sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df_kb["customer_text"].fillna(""))
        self.kb_records = self.df_kb[["conversation_id", "customer_text", "support_reply"]].to_dict("records")
        
        try:
            logger.info("Caching index to %s...", INDEX_CACHE_PATH)
            tmp_path = INDEX_CACHE_PATH.with_suffix(".tmp")
            with open(tmp_path, "wb") as f:
                pickle.dump({
                    "df_kb": self.df_kb,
                    "vectorizer": self.vectorizer,
                    "tfidf_matrix": self.tfidf_matrix,
                    "kb_records": self.kb_records
                }, f)
            import os
            os.replace(tmp_path, INDEX_CACHE_PATH)
            logger.info("Index built and cached successfully.")
        except Exception as exc:
            logger.warning("Could not cache index to %s (%s). Index remains active in memory.", INDEX_CACHE_PATH, exc)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves top_k most similar historical customer->brand resolution cases.
        Uses pre-indexed native dictionaries for O(1) zero-allocation lookup.
        """
        if not query or self.vectorizer is None:
            return []
            
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        top_indices = sims.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            score = float(sims[idx])
            record = self.kb_records[idx] if self.kb_records is not None else self.df_kb.iloc[idx]
            results.append({
                "conversation_id": record["conversation_id"],
                "customer_text": mask_pii(str(record["customer_text"])),
                "support_reply": mask_pii(str(record["support_reply"])),
                "similarity_score": round(score, 4)
            })
        return results

if __name__ == "__main__":
    retriever = HistoricalRetriever()
    test_query = "My package says delivered yesterday but it is not on my porch. Where is my stuff?"
    hits = retriever.retrieve(test_query, top_k=3)
    print(f"\nQuery: {test_query}")
    print("Top Retrieved Historical Cases:")
    for i, h in enumerate(hits, 1):
        print(f"\n[{i}] Similarity: {h['similarity_score']} | Conv ID: {h['conversation_id']}")
        print(f"    Historical Customer: {h['customer_text']}")
        print(f"    Historical Reply   : {h['support_reply']}")
