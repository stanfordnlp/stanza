""" see __init__.py """

from datetime import datetime
import os
import pickle
import random
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np      # type: ignore
import jsonlines        # type: ignore
import toml
import torch
from tqdm import tqdm   # type: ignore
import transformers     # type: ignore

from coref import bert, conll, utils
from coref.anaphoricity_scorer import AnaphoricityScorer
from coref.cluster_checker import ClusterChecker
from coref.config import Config
from coref.const import CorefResult, Doc
from coref.loss import CorefLoss
from coref.pairwise_encoder import PairwiseEncoder
from coref.rough_scorer import RoughScorer
from coref.span_predictor import SpanPredictor
from coref.tokenizer_customization import TOKENIZER_FILTERS, TOKENIZER_MAPS
from coref.utils import GraphNode
from coref.word_encoder import WordEncoder


class CorefModel:  # pylint: disable=too-many-instance-attributes
    """Combines all coref modules together to find coreferent spans.

    Attributes:
        config (coref.config.Config): the model's configuration,
            see config.toml for the details
        epochs_trained (int): number of epochs the model has been trained for
        trainable (Dict[str, torch.nn.Module]): trainable submodules with their
            names used as keys
        training (bool): used to toggle train/eval modes

    Submodules (in the order of their usage in the pipeline):
        tokenizer (transformers.AutoTokenizer)
        bert (transformers.AutoModel)
        we (WordEncoder)
        rough_scorer (RoughScorer)
        pw (PairwiseEncoder)
        a_scorer (AnaphoricityScorer)
        sp (SpanPredictor)
    """
    def __init__(self,
                 config_path: str,
                 section: str,
                 epochs_trained: int = 0,
                 build_optimizers: bool = True):
        """
        A newly created model is set to evaluation mode.

        Args:
            config_path (str): the path to the toml file with the configuration
            section (str): the selected section of the config file
            epochs_trained (int): the number of epochs finished
                (useful for warm start)
        """
        self.config = CorefModel._load_config(config_path, section)
        self.epochs_trained = epochs_trained
        self._docs: Dict[str, List[Doc]] = {}
        self._build_model()
        if build_optimizers:
            self._build_optimizers()
        self._set_training(False)
        self._coref_criterion = CorefLoss(self.config.bce_loss_weight)
        self._span_criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    @property
    def training(self) -> bool:
        """ Represents whether the model is in the training mode """
        return self._training

    @training.setter
    def training(self, new_value: bool):
        if self._training is new_value:
            return
        self._set_training(new_value)

    # ========================================================== Public methods

    @torch.no_grad()
    def evaluate(self,
                 data_split: str = "dev",
                 word_level_conll: bool = False
                 ) -> Tuple[float, Tuple[float, float, float]]:
        """ Evaluates the modes on the data split provided.

        Args:
            data_split (str): one of 'dev'/'test'/'train'
            word_level_conll (bool): if True, outputs conll files on word-level

        Returns:
            mean loss
            span-level LEA: f1, precision, recal
        """
        self.training = False
        w_checker = ClusterChecker()
        s_checker = ClusterChecker()
        docs = self._get_docs(self.config.__dict__[f"{data_split}_data"])
        running_loss = 0.0
        s_correct = 0
        s_total = 0

        with conll.open_(self.config, self.epochs_trained, data_split) \
                as (gold_f, pred_f):
            pbar = tqdm(docs, unit="docs", ncols=0)
            for doc in pbar:
                res = self.run(doc)

                running_loss += self._coref_criterion(res.coref_scores, res.coref_y).item()

                if res.span_y:
                    pred_starts = res.span_scores[:, :, 0].argmax(dim=1)
                    pred_ends = res.span_scores[:, :, 1].argmax(dim=1)
                    s_correct += ((res.span_y[0] == pred_starts) * (res.span_y[1] == pred_ends)).sum().item()
                    s_total += len(pred_starts)

                if word_level_conll:
                    conll.write_conll(doc,
                                      [[(i, i + 1) for i in cluster]
                                       for cluster in doc["word_clusters"]],
                                      gold_f)
                    conll.write_conll(doc,
                                      [[(i, i + 1) for i in cluster]
                                       for cluster in res.word_clusters],
                                      pred_f)
                else:
                    conll.write_conll(doc, doc["span_clusters"], gold_f)
                    conll.write_conll(doc, res.span_clusters, pred_f)

                w_checker.add_predictions(doc["word_clusters"], res.word_clusters)
                w_lea = w_checker.total_lea

                s_checker.add_predictions(doc["span_clusters"], res.span_clusters)
                s_lea = s_checker.total_lea

                del res

                pbar.set_description(
                    f"{data_split}:"
                    f" | WL: "
                    f" loss: {running_loss / (pbar.n + 1):<.5f},"
                    f" f1: {w_lea[0]:.5f},"
                    f" p: {w_lea[1]:.5f},"
                    f" r: {w_lea[2]:<.5f}"
                    f" | SL: "
                    f" sa: {s_correct / s_total:<.5f},"
                    f" f1: {s_lea[0]:.5f},"
                    f" p: {s_lea[1]:.5f},"
                    f" r: {s_lea[2]:<.5f}"
                )
            print()

        return (running_loss / len(docs), *s_checker.total_lea)

    def load_weights(self,
                     path: Optional[str] = None,
                     ignore: Optional[Set[str]] = None,
                     map_location: Optional[str] = None,
                     noexception: bool = False) -> None:
        """
        Loads pretrained weights of modules saved in a file located at path.
        If path is None, the last saved model with current configuration
        in data_dir is loaded.
        Assumes files are named like {configuration}_(e{epoch}_{time})*.pt.
        """
        if path is None:
            pattern = rf"{self.config.section}_\(e(\d+)_[^()]*\).*\.pt"
            files = []
            for f in os.listdir(self.config.data_dir):
                match_obj = re.match(pattern, f)
                if match_obj:
                    files.append((int(match_obj.group(1)), f))
            if not files:
                if noexception:
                    print("No weights have been loaded", flush=True)
                    return
                raise OSError(f"No weights found in {self.config.data_dir}!")
            _, path = sorted(files)[-1]
            path = os.path.join(self.config.data_dir, path)

        if map_location is None:
            map_location = self.config.device
        print(f"Loading from {path}...")
        state_dicts = torch.load(path, map_location=map_location)
        self.epochs_trained = state_dicts.pop("epochs_trained", 0)
        for key, state_dict in state_dicts.items():
            if not ignore or key not in ignore:
                if key.endswith("_optimizer"):
                    self.optimizers[key].load_state_dict(state_dict)
                elif key.endswith("_scheduler"):
                    self.schedulers[key].load_state_dict(state_dict)
                else:
                    self.trainable[key].load_state_dict(state_dict)
                print(f"Loaded {key}")

    def run(self,  # pylint: disable=too-many-locals
            doc: Doc,
            ) -> CorefResult:
        """
        This is a massive method, but it made sense to me to not split it into
        several ones to let one see the data flow.

        Args:
            doc (Doc): a dictionary with the document data.

        Returns:
            CorefResult (see const.py)
        """
        # Encode words with bert
        # words           [n_words, span_emb]
        # cluster_ids     [n_words]
        words, cluster_ids = self.we(doc, self._bertify(doc))

        # Obtain bilinear scores and leave only top-k antecedents for each word
        # top_rough_scores  [n_words, n_ants]
        # top_indices       [n_words, n_ants]
        top_rough_scores, top_indices = self.rough_scorer(words)

        # Get pairwise features [n_words, n_ants, n_pw_features]
        pw = self.pw(top_indices, doc)

        batch_size = self.config.a_scoring_batch_size
        a_scores_lst: List[torch.Tensor] = []

        for i in range(0, len(words), batch_size):
            pw_batch = pw[i:i + batch_size]
            words_batch = words[i:i + batch_size]
            top_indices_batch = top_indices[i:i + batch_size]
            top_rough_scores_batch = top_rough_scores[i:i + batch_size]

            # a_scores_batch    [batch_size, n_ants]
            a_scores_batch = self.a_scorer(
                all_mentions=words, mentions_batch=words_batch,
                pw_batch=pw_batch, top_indices_batch=top_indices_batch,
                top_rough_scores_batch=top_rough_scores_batch
            )
            a_scores_lst.append(a_scores_batch)

        res = CorefResult()

        # coref_scores  [n_spans, n_ants]
        res.coref_scores = torch.cat(a_scores_lst, dim=0)

        res.coref_y = self._get_ground_truth(
            cluster_ids, top_indices, (top_rough_scores > float("-inf")))
        res.word_clusters = self._clusterize(doc, res.coref_scores,
                                             top_indices)
        res.span_scores, res.span_y = self.sp.get_training_data(doc, words)

        if not self.training:
            res.span_clusters = self.sp.predict(doc, words, res.word_clusters)

        return res

    def save_weights(self):
        """ Saves trainable models as state dicts. """
        to_save: List[Tuple[str, Any]] = \
            [(key, value) for key, value in self.trainable.items()
             if self.config.bert_finetune or key != "bert"]
        to_save.extend(self.optimizers.items())
        to_save.extend(self.schedulers.items())

        time = datetime.strftime(datetime.now(), "%Y.%m.%d_%H.%M")
        path = os.path.join(self.config.data_dir,
                            f"{self.config.section}"
                            f"_(e{self.epochs_trained}_{time}).pt")
        savedict = {name: module.state_dict() for name, module in to_save}
        savedict["epochs_trained"] = self.epochs_trained  # type: ignore
        torch.save(savedict, path)

    def train(self):
        """
        Trains all the trainable blocks in the model using the config provided.
        """
        docs = list(self._get_docs(self.config.train_data))
        docs_ids = list(range(len(docs)))
        avg_spans = sum(len(doc["head2span"]) for doc in docs) / len(docs)

        for epoch in range(self.epochs_trained, self.config.train_epochs):
            self.training = True
            running_c_loss = 0.0
            running_s_loss = 0.0
            random.shuffle(docs_ids)
            pbar = tqdm(docs_ids, unit="docs", ncols=0)
            for doc_id in pbar:
                doc = docs[doc_id]

                for optim in self.optimizers.values():
                    optim.zero_grad()

                res = self.run(doc)

                c_loss = self._coref_criterion(res.coref_scores, res.coref_y)
                if res.span_y:
                    s_loss = (self._span_criterion(res.span_scores[:, :, 0], res.span_y[0])
                              + self._span_criterion(res.span_scores[:, :, 1], res.span_y[1])) / avg_spans / 2
                else:
                    s_loss = torch.zeros_like(c_loss)

                del res

                (c_loss + s_loss).backward()
                running_c_loss += c_loss.item()
                running_s_loss += s_loss.item()

                del c_loss, s_loss

                for optim in self.optimizers.values():
                    optim.step()
                for scheduler in self.schedulers.values():
                    scheduler.step()

                pbar.set_description(
                    f"Epoch {epoch + 1}:"
                    f" {doc['document_id']:26}"
                    f" c_loss: {running_c_loss / (pbar.n + 1):<.5f}"
                    f" s_loss: {running_s_loss / (pbar.n + 1):<.5f}"
                )

            self.epochs_trained += 1
            self.save_weights()
            self.evaluate()

    # ========================================================= Private methods

    def _bertify(self, doc: Doc) -> torch.Tensor:
        subwords_batches = bert.get_subwords_batches(doc, self.config,
                                                     self.tokenizer)

        special_tokens = np.array([self.tokenizer.cls_token_id,
                                   self.tokenizer.sep_token_id,
                                   self.tokenizer.pad_token_id])
        subword_mask = ~(np.isin(subwords_batches, special_tokens))

        subwords_batches_tensor = torch.tensor(subwords_batches,
                                               device=self.config.device,
                                               dtype=torch.long)
        subword_mask_tensor = torch.tensor(subword_mask,
                                           device=self.config.device)

        # Obtain bert output for selected batches only
        attention_mask = (subwords_batches != self.tokenizer.pad_token_id)
        out = self.bert(
            subwords_batches_tensor,
            attention_mask=torch.tensor(
                attention_mask, device=self.config.device))
        out = out['last_hidden_state']

        # [n_subwords, bert_emb]
        return out[subword_mask_tensor]

    def _build_model(self):
        self.bert, self.tokenizer = bert.load_bert(self.config)
        self.pw = PairwiseEncoder(self.config).to(self.config.device)

        bert_emb = self.bert.config.hidden_size
        pair_emb = bert_emb * 3 + self.pw.shape

        # pylint: disable=line-too-long
        self.a_scorer = AnaphoricityScorer(pair_emb, self.config).to(self.config.device)
        self.we = WordEncoder(bert_emb, self.config).to(self.config.device)
        self.rough_scorer = RoughScorer(bert_emb, self.config).to(self.config.device)
        self.sp = SpanPredictor(bert_emb, self.config.sp_embedding_size).to(self.config.device)

        self.trainable: Dict[str, torch.nn.Module] = {
            "bert": self.bert, "we": self.we,
            "rough_scorer": self.rough_scorer,
            "pw": self.pw, "a_scorer": self.a_scorer,
            "sp": self.sp
        }

    def _build_optimizers(self):
        n_docs = len(self._get_docs(self.config.train_data))
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.schedulers: Dict[str, torch.optim.lr_scheduler.LambdaLR] = {}

        for param in self.bert.parameters():
            param.requires_grad = self.config.bert_finetune

        if self.config.bert_finetune:
            self.optimizers["bert_optimizer"] = torch.optim.Adam(
                self.bert.parameters(), lr=self.config.bert_learning_rate
            )
            self.schedulers["bert_scheduler"] = \
                transformers.get_linear_schedule_with_warmup(
                    self.optimizers["bert_optimizer"],
                    n_docs, n_docs * self.config.train_epochs
                )

        # Must ensure the same ordering of parameters between launches
        modules = sorted((key, value) for key, value in self.trainable.items()
                         if key != "bert")
        params = []
        for _, module in modules:
            for param in module.parameters():
                param.requires_grad = True
                params.append(param)

        self.optimizers["general_optimizer"] = torch.optim.Adam(
            params, lr=self.config.learning_rate)
        self.schedulers["general_scheduler"] = \
            transformers.get_linear_schedule_with_warmup(
                self.optimizers["general_optimizer"],
                0, n_docs * self.config.train_epochs
            )

    def _clusterize(self, doc: Doc, scores: torch.Tensor, top_indices: torch.Tensor):
        antecedents = scores.argmax(dim=1) - 1
        not_dummy = antecedents >= 0
        coref_span_heads = torch.arange(0, len(scores))[not_dummy]
        antecedents = top_indices[coref_span_heads, antecedents[not_dummy]]

        nodes = [GraphNode(i) for i in range(len(doc["cased_words"]))]
        for i, j in zip(coref_span_heads.tolist(), antecedents.tolist()):
            nodes[i].link(nodes[j])
            assert nodes[i] is not nodes[j]

        clusters = []
        for node in nodes:
            if len(node.links) > 0 and not node.visited:
                cluster = []
                stack = [node]
                while stack:
                    current_node = stack.pop()
                    current_node.visited = True
                    cluster.append(current_node.id)
                    stack.extend(link for link in current_node.links if not link.visited)
                assert len(cluster) > 1
                clusters.append(sorted(cluster))
        return sorted(clusters)

    def _get_docs(self, path: str) -> List[Doc]:
        if path not in self._docs:
            basename = os.path.basename(path)
            model_name = self.config.bert_model.replace("/", "_")
            cache_filename = f"{model_name}_{basename}.pickle"
            if os.path.exists(cache_filename):
                with open(cache_filename, mode="rb") as cache_f:
                    self._docs[path] = pickle.load(cache_f)
            else:
                self._docs[path] = self._tokenize_docs(path)
                with open(cache_filename, mode="wb") as cache_f:
                    pickle.dump(self._docs[path], cache_f)
        return self._docs[path]

    @staticmethod
    def _get_ground_truth(cluster_ids: torch.Tensor,
                          top_indices: torch.Tensor,
                          valid_pair_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cluster_ids: tensor of shape [n_words], containing cluster indices
                for each word. Non-gold words have cluster id of zero.
            top_indices: tensor of shape [n_words, n_ants],
                indices of antecedents of each word
            valid_pair_map: boolean tensor of shape [n_words, n_ants],
                whether for pair at [i, j] (i-th word and j-th word)
                j < i is True

        Returns:
            tensor of shape [n_words, n_ants + 1] (dummy added),
                containing 1 at position [i, j] if i-th and j-th words corefer.
        """
        y = cluster_ids[top_indices] * valid_pair_map  # [n_words, n_ants]
        y[y == 0] = -1                                 # -1 for non-gold words
        y = utils.add_dummy(y)                         # [n_words, n_cands + 1]
        y = (y == cluster_ids.unsqueeze(1))            # True if coreferent
        # For all rows with no gold antecedents setting dummy to True
        y[y.sum(dim=1) == 0, 0] = True
        return y.to(torch.float)

    @staticmethod
    def _load_config(config_path: str,
                     section: str) -> Config:
        config = toml.load(config_path)
        default_section = config["DEFAULT"]
        current_section = config[section]
        unknown_keys = (set(current_section.keys())
                        - set(default_section.keys()))
        if unknown_keys:
            raise ValueError(f"Unexpected config keys: {unknown_keys}")
        return Config(section, **{**default_section, **current_section})

    def _set_training(self, value: bool):
        self._training = value
        for module in self.trainable.values():
            module.train(self._training)

    def _tokenize_docs(self, path: str) -> List[Doc]:
        print(f"Tokenizing documents at {path}...", flush=True)
        out: List[Doc] = []
        filter_func = TOKENIZER_FILTERS.get(self.config.bert_model,
                                            lambda _: True)
        token_map = TOKENIZER_MAPS.get(self.config.bert_model, {})
        with jsonlines.open(path, mode="r") as data_f:
            for doc in data_f:
                doc["span_clusters"] = [[tuple(mention) for mention in cluster]
                                   for cluster in doc["span_clusters"]]
                word2subword = []
                subwords = []
                word_id = []
                for i, word in enumerate(doc["cased_words"]):
                    tokenized_word = (token_map[word]
                                      if word in token_map
                                      else self.tokenizer.tokenize(word))
                    tokenized_word = list(filter(filter_func, tokenized_word))
                    word2subword.append((len(subwords), len(subwords) + len(tokenized_word)))
                    subwords.extend(tokenized_word)
                    word_id.extend([i] * len(tokenized_word))
                doc["word2subword"] = word2subword
                doc["subwords"] = subwords
                doc["word_id"] = word_id
                out.append(doc)
        print("Tokenization OK", flush=True)
        return out
