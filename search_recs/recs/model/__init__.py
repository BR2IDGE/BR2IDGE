__all__ = [
    "BaseRecsModel",
    "DeepFMModel",
    "LightFMModel",
    "Bert4REC",
    "ALSModel",
    "NMFRecommender",
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
    if name == "Bert4REC":
        from .bert4rec import Bert4REC
        return Bert4REC
    if name == "ALSModel":
        from .als_model import ALSModel
        return ALSModel
    if name == "NMFRecommender":
        from .nmf_recommender import NMFRecommender
        return NMFRecommender
    raise AttributeError(name)
