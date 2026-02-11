from .base_dataloader import BuildConfig
from .goodbooks import load_goodbooks_dataset
from .movielens import load_movielens_dataset, load_movielens_user_query_dataset, load_genome_tag_map
from .generic_csv import load_generic_csv_dataset
from .amazonEletronics import get_loader as load_amazon_search_dataset

from .lastfm import load_lastfm_dataset as load_lastfm_dataset

REGISTRY = {
    "goodbooks": load_goodbooks_dataset,
    "movielens": load_movielens_dataset,
    "lastfm": load_lastfm_dataset,  
    "generic_csv": load_generic_csv_dataset,
    "amazoneletronics": load_amazon_search_dataset
}

def get_loader(name: str):
    name = name.lower()
    if name not in REGISTRY:
        raise ValueError(f"Dataset '{name}' inválido. Opções: {list(REGISTRY)}")
    return REGISTRY[name]