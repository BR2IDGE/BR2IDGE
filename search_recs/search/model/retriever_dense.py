from typing import Optional, Tuple, List, Dict, Set, Iterable, Any
import pandas as pd
from retriv import DenseRetriever
import nltk
import retriv.base_retriever

from search_recs.recs.model import BaseRecsModel

def fixed_map_internal_ids_to_original_ids(self, doc_ids: Iterable) -> List[str]:
    """
    Patched method to filter out -1 (Faiss padding) before mapping to original IDs.
    Prevents KeyError when retriv tries to map non-existent ID -1.
    """
    return [self.id_mapping[doc_id] for doc_id in doc_ids if doc_id != -1]

print("[DenseRetrieverModel] Applying patch to BaseRetriever to fix KeyError: -1...")
retriv.base_retriever.BaseRetriever.map_internal_ids_to_original_ids = fixed_map_internal_ids_to_original_ids


class DenseRetrieverModel(BaseRecsModel):
    def __init__(self, model_config: dict):
        super().__init__(model_config)
        params = self.model_params or {}

        self.index_name: str = params.get("index_name", "dense_index")
        self.model: str = params.get("model", "sentence-transformers/all-MiniLM-L6-v2")

        self.qcol: str = params.get("query_col", "search_query")
        self.dcol: str = params.get("doc_col", "document")
        self.ccol: str = params.get("cat_col", "category")

        self.retriever_params: Dict = params.get("retriever_params", {
            "normalize": True,
            "max_length": 256,
            "use_ann": True,
        })
        
        self.index_params: Dict = params.get("index_params", {
            "use_gpu": True,
            "batch_size": 512,
            "show_progress": True,
        })

        self._collection: Optional[List[Dict]] = None
        self._dr: Optional[DenseRetriever] = None
        self._doc_to_id: Optional[Dict[str, int]] = None
        
        self._train_df: Optional[pd.DataFrame] = None
        self._test_df: Optional[pd.DataFrame] = None
        self._val_df: Optional[pd.DataFrame] = None
        
        self.corpus_size: int = 0

        print(f"[DenseRetrieverModel] Init with index_name={self.index_name}, model={self.model}")

    def _build_collection(self, df: pd.DataFrame) -> List[Dict]:
        """Converts unique Document DataFrame into a list of dicts [{id, text}]."""
        collection_df = df.rename(columns={self.dcol: "text"}).reset_index(drop=True)
        collection_df['id'] = collection_df.index
        
        self._doc_to_id = pd.Series(collection_df.id.values, index=collection_df.text).to_dict()
        
        return collection_df[["id", "text"]].to_dict("records")

    def preprocess(self, train_data: pd.DataFrame, **kwargs):
        """
        Prepares the collection (corpus) using all available unique documents 
        (Train + Val + Test) to ensure a complete search space.
        """
        if not isinstance(train_data, pd.DataFrame):
            raise ValueError(f"train_data must be a DataFrame with columns [{self.qcol}, {self.dcol}].")
        
        self._train_df = train_data
        self._test_df = kwargs.get("test_data")
        self._val_df = kwargs.get("val_data")
        
        doc_sources = [train_data[[self.dcol]]]
        
        if self._val_df is not None:
            doc_sources.append(self._val_df[[self.dcol]])
        
        if self._test_df is not None:
            doc_sources.append(self._test_df[[self.dcol]])

        all_docs_df = pd.concat(doc_sources, ignore_index=True)

        unique_docs_df = all_docs_df.drop_duplicates().reset_index(drop=True)

        self._collection = self._build_collection(unique_docs_df)
        self.corpus_size = len(self._collection)

        print(f"[DenseRetrieverModel] Preprocess complete — Global collection created with {self.corpus_size} unique docs.")

    def fit(self):
        """Builds the dense vector index."""
        if self._collection is None:
            raise RuntimeError("Execute preprocess() before fit().")
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)

        DenseRetriever.delete(self.index_name)
        self._dr = DenseRetriever(
            index_name=self.index_name,
            model=self.model,
            **self.retriever_params
        )
        self._dr.index(self._collection, **self.index_params)
        print(f"[DenseRetrieverModel] Fit complete — Collection indexed ({self.corpus_size} docs).")

    def prediction(self, test_data: pd.DataFrame) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Generates predictions by comparing every test query against the ENTIRE indexed corpus.
        Aligns y_true and y_pred vectors with the global document universe.
        """
        if self._dr is None:
            raise RuntimeError("Execute fit() before prediction().")

        if not isinstance(test_data, pd.DataFrame):
            raise ValueError(f"test_data must be a DataFrame with columns [{self.qcol}, {self.dcol}].")

        all_docs_ids = [doc['id'] for doc in self._collection]
        
        if self._doc_to_id is None:
            raise RuntimeError("Internal Error: _doc_to_id is None. Did preprocess() run?")
        
        if self.corpus_size == 0:
            raise RuntimeError("Empty Corpus. Run preprocess() with valid data.")

        relevance_map_ids: Dict[str, Set[int]] = {}
        
        for _, row in test_data.iterrows():
            q = row[self.qcol]
            d_text = row[self.dcol]
            
            d_id = self._doc_to_id.get(d_text)
            
            if d_id is not None:
                if q not in relevance_map_ids:
                    relevance_map_ids[q] = set()
                relevance_map_ids[q].add(d_id)

        unique_queries = test_data[self.qcol].unique()
        
        queries_for_search = [
            {"id": str(qid), "text": qtext} for qid, qtext in enumerate(unique_queries)
        ]

        print(f"[DenseRetrieverModel] Searching {len(unique_queries)} unique queries against {self.corpus_size} docs...")
        
        if queries_for_search:
            batch_results = self._dr.bsearch(queries_for_search, cutoff=self.corpus_size)
        else:
            batch_results = {}
        
        query_to_results = {}

        for qid, qtext in enumerate(unique_queries):
            hits = batch_results.get(str(qid), {}) 

            scores_dict = {}
            for doc_id, score in hits.items():
                scores_dict[int(doc_id)] = float(score)

            y_pred = [scores_dict.get(did, 0.0) for did in all_docs_ids]
            
            relevant_ids = relevance_map_ids.get(qtext, set())
            y_true = [1 if did in relevant_ids else 0 for did in all_docs_ids]

            query_to_results[qtext] = (y_true, y_pred)

        final_y_true = []
        final_y_pred = []
        for q_text in test_data[self.qcol].tolist():
            y_t, y_p = query_to_results[q_text]
            final_y_true.append(y_t)
            final_y_pred.append(y_p)
        
        print(f"[DenseRetrieverModel] Prediction complete — {len(final_y_true)} predictions generated.")
        return final_y_true, final_y_pred
    
    def search(self, queries: List[str], k: int = 10) -> Dict[str, List[Dict]]:
        """
        Performs a search for a batch of queries and returns top-k ranking for each.

        Args:
            queries (List[str]): List of query strings.
            k (int): Number of documents to return per query (cutoff).

        Returns:
            Dict[str, List[Dict]]: Dictionary where key is the original query (str)
                                   and value is a list of results:
                                   [{"doc_text": "...", "score": 0.85}, ...]
        """
        if self._dr is None:
            raise RuntimeError("Call fit() before search(). Index must be created first.")

        queries_for_search = [
            {"id": str(qid), "text": qtext} for qid, qtext in enumerate(queries)
        ]

        print(f"[DenseRetrieverModel] Searching {len(queries)} queries with k={k}...")
        batch_results = self._dr.bsearch(queries_for_search, cutoff=k)

        id_to_doc = {item["id"]: item["text"] for item in self._collection}

        results: Dict[str, List[Dict]] = {}

        for qid, qtext in enumerate(queries):
            hits = batch_results.get(str(qid), {})
            query_results: List[Dict] = []

            if isinstance(hits, dict):
                iterator = hits.items()
            elif isinstance(hits, list):
                iterator = hits
            else:
                iterator = []

            for item in iterator:
                doc_id = None
                score = 0.0

                if isinstance(item, (tuple, list)) and len(item) == 2:
                    doc_id, score = item
                elif isinstance(item, int):
                    doc_id = item
                    score = 1.0
                elif isinstance(hits, dict):
                    doc_id, score = item
                
                if doc_id is not None:
                    doc_text = id_to_doc.get(int(doc_id), f"[Doc ID {doc_id} not found]")
                    query_results.append({
                        "doc_text": doc_text,
                        "score": float(score)
                    })

            results[qtext] = query_results
        
        print(f"[DenseRetrieverModel] Search complete.") 
        return results