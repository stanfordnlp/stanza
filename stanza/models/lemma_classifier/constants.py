from torchtext.vocab import GloVe

def get_glove(embedding_dim: int) -> GloVe:
    # Give back GloVe embeddings
    return GloVe(name='6B', dim=embedding_dim)

UNKNOWN_TOKEN = "unk"  # token name for unknown tokens
UNKNOWN_TOKEN_IDX = -1   # custom index we apply to unknown tokens