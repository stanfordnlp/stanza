from torchtext.vocab import GloVe

def get_glove(embedding_dim: int) -> GloVe:
    return GloVe(name='6B', dim=embedding_dim)

UNKNOWN_TOKEN = "unk"
UNKNOWN_TOKEN_IDX = -1