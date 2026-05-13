__all__ = [
    "BaseRecsModel",
    "DeepFMModel",
    "LightFMModel",
    "LightGCNModel",
    "Bert4REC",
    "ALSModel",
    "NMFRecommender",
    "ItemKNNModel",
]

def __getattr__(name):
    if name == "BaseRecsModel":
        from .base_model import BaseRecsModel
        return BaseRecsModel
    if name == "LightFMModel":
        from .light_fm import LightFMModel
        return LightFMModel
    if name == "DeepFMModel":
        from .libreco_model import DeepFMModel
        return DeepFMModel
    if name == "LightGCNModel":
        from .light_gcn import LightGCNModel
        return LightGCNModel
    if name == "Bert4REC":
        from .bert4rec import Bert4REC
        return Bert4REC
    if name == "ALSModel":
        from .als_model import ALSModel
        return ALSModel
    if name == "NMFRecommender":
        from .nmf_recommender import NMFRecommender
        return NMFRecommender
    if name == "ItemKNNModel":
        from .item_knn import ItemKNNModel
        return ItemKNNModel
    raise AttributeError(name)
