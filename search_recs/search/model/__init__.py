__all__ = [
    "BiEncoderSearchModel",
    "BM25Model",
    "DenseRetrieverModel",
]

def __getattr__(name):
    if name == "BiEncoderSearchModel":
        from .bi_encoder import BiEncoderModel
        return BiEncoderModel
    if name == "BM25Model":
        from .bm25_sparse import BM25Model
        return BM25Model
    if name == "DenseRetrieverModel":
        from .retriever_dense import DenseRetrieverModel
        return DenseRetrieverModel
    raise AttributeError(name)
