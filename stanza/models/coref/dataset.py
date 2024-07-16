import json
import logging
from torch.utils.data import Dataset

from stanza.models.coref.tokenizer_customization import TOKENIZER_FILTERS, TOKENIZER_MAPS

logger = logging.getLogger('stanza')

class CorefDataset(Dataset):

    def __init__(self, path, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        # by default, this doesn't filter anything (see lambda _ True);
        # however, there are some subword symbols which are standalone
        # tokens which we don't want on models like Albert; hence we
        # pass along a filter if needed.
        self.__filter_func = TOKENIZER_FILTERS.get(self.config.bert_model,
                                                   lambda _: True)
        self.__token_map = TOKENIZER_MAPS.get(self.config.bert_model, {})

        try:
            with open(path, encoding="utf-8") as fin:
                data_f = json.load(fin)
        except json.decoder.JSONDecodeError:
            # read the old jsonlines format if necessary
            with open(path, encoding="utf-8") as fin:
                text = "[" + ",\n".join(fin) + "]"
            data_f = json.loads(text)
        logger.info("Processing %d docs from %s...", len(data_f), path)
        self.__raw = data_f
        self.__avg_span = sum(len(doc["head2span"]) for doc in self.__raw) / len(self.__raw)
        self.__out = []
        for doc in self.__raw:
            doc["span_clusters"] = [[tuple(mention) for mention in cluster]
                                for cluster in doc["span_clusters"]]
            word2subword = []
            subwords = []
            word_id = []
            for i, word in enumerate(doc["cased_words"]):
                tokenized_word = self.__token_map.get(word, self.tokenizer.tokenize(word))
                tokenized_word = list(filter(self.__filter_func, tokenized_word))
                word2subword.append((len(subwords), len(subwords) + len(tokenized_word)))
                subwords.extend(tokenized_word)
                word_id.extend([i] * len(tokenized_word))
            doc["word2subword"] = word2subword
            doc["subwords"] = subwords
            doc["word_id"] = word_id
            self.__out.append(doc)
        logger.info("Loaded %d docs from %s.", len(data_f), path)

    @property
    def avg_span(self):
        return self.__avg_span

    def __getitem__(self, x):
        return self.__out[x]

    def __len__(self):
        return len(self.__out)
