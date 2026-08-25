from search_recs.recs.dataloader.recs_dataloader import RecsDataLoader
from search_recs.recs.dataloader.movielens import MovieLens25MDataLoader
from search_recs.recs.dataloader.lastfm import LastFM360KDataLoader
from search_recs.recs.dataloader.generic import GenericRecsDataLoader
from search_recs.recs.dataloader.amazonEletronics import AmazonElectronicsDataLoader
from search_recs.recs.dataloader.retrieval_as_user import RetrievalAsUserDataLoader
from search_recs.recs.dataloader.beir import BeirQueryAsUserDataLoader

__all__ = [
    "RecsDataLoader",
    "MovieLens25MDataLoader",
    "LastFM360KDataLoader",
    "GenericRecsDataLoader",
    "AmazonElectronicsDataLoader",
    "RetrievalAsUserDataLoader",
    "BeirQueryAsUserDataLoader",
]

REGISTRY = {
    "movielens": MovieLens25MDataLoader,
    "lastfm": LastFM360KDataLoader,
    "generic": GenericRecsDataLoader,
    "amazonElectronics": AmazonElectronicsDataLoader,
    "retrieval_as_user": RetrievalAsUserDataLoader,
}

from search_recs.datasets.manager import BEIR_SUBSETS  


def _beir_recs_loader(subset: str):
    def _factory(full_config: dict):
        cfg = dict(full_config)
        dl = dict(cfg.get("dataloader", cfg))
        dl.setdefault("subset", subset)
        if "dataloader" in cfg:
            cfg["dataloader"] = dl
        else:
            cfg = dl
        return BeirQueryAsUserDataLoader(cfg)
    return _factory


for _subset in BEIR_SUBSETS:
    REGISTRY[f"beir_{_subset}"] = _beir_recs_loader(_subset)

def get_loader(name: str):
    if name in REGISTRY:
        return REGISTRY[name]
    
    if name.lower() in REGISTRY:
        return REGISTRY[name.lower()]
    
    for cls in REGISTRY.values():
        if cls.__name__ == name:
            return cls
            
    raise ValueError(f"DataLoader '{name}' not found. Options: {list(REGISTRY.keys())}")