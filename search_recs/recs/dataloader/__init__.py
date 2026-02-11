from search_recs.recs.dataloader.recs_dataloader import RecsDataLoader
from search_recs.recs.dataloader.movielens import MovieLens25MDataLoader
from search_recs.recs.dataloader.lastfm import LastFM360KDataLoader
from search_recs.recs.dataloader.generic import GenericRecsDataLoader
from search_recs.recs.dataloader.amazonEletronics import AmazonElectronicsDataLoader
from search_recs.recs.dataloader.retrieval_as_user import RetrievalAsUserDataLoader

__all__ = [
    "RecsDataLoader",
    "MovieLens25MDataLoader",
    "LastFM360KDataLoader",
    "GenericRecsDataLoader",
    "AmazonElectronicsDataLoader",
    "RetrievalAsUserDataLoader"
]

REGISTRY = {
    "movielens": MovieLens25MDataLoader,
    "lastfm": LastFM360KDataLoader,
    "generic": GenericRecsDataLoader,
    "amazonElectronics": AmazonElectronicsDataLoader,
    "retrieval_as_user": RetrievalAsUserDataLoader,
}

def get_loader(name: str):
    """Returns the DataLoader class based on the given name."""
    if name in REGISTRY:
        return REGISTRY[name]
    
    if name.lower() in REGISTRY:
        return REGISTRY[name.lower()]
    
    # Check if 'name' matches the class name directly
    for cls in REGISTRY.values():
        if cls.__name__ == name:
            return cls
            
    raise ValueError(f"DataLoader '{name}' not found. Options: {list(REGISTRY.keys())}")