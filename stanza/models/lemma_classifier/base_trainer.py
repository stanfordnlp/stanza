
from abc import ABC, abstractmethod
import logging
import os
from typing import List, Tuple, Any, Mapping

import torch
import torch.nn as nn
import torch.optim as optim

from stanza.models.common.utils import default_device
from stanza.models.lemma_classifier import utils
from stanza.models.lemma_classifier.constants import DEFAULT_BATCH_SIZE
from stanza.models.lemma_classifier.evaluate_models import evaluate_model
from stanza.utils.get_tqdm import get_tqdm

tqdm = get_tqdm()
logger = logging.getLogger('stanza.lemmaclassifier')

class BaseLemmaClassifierTrainer(ABC):
    def configure_weighted_loss(self, label_decoder: Mapping, counts: Mapping):
        """
        If applicable, this function will update the loss function of the LemmaClassifierLSTM model to become BCEWithLogitsLoss.
        The weights are determined by the counts of the classes in the dataset. The weights are inversely proportional to the
        frequency of the class in the set. E.g. classes with lower frequency will have higher weight.
        """
        weights = [0 for _ in label_decoder.keys()]  # each key in the label decoder is one class, we have one weight per class
        total_samples = sum(counts.values())
        for class_idx in counts:
            weights[class_idx] = total_samples / (counts[class_idx] * len(counts))  # weight_i = total / (# examples in class i * num classes)
        weights = torch.tensor(weights)
        logger.info(f"Using weights {weights} for weighted loss.")
        self.criterion = nn.BCEWithLogitsLoss(weight=weights)

    @abstractmethod
    def build_model(self, label_decoder, upos_to_id, known_words, target_words, target_upos):
        """
        Build a model using pieces of the dataset to determine some of the model shape
        """

    def train(self, num_epochs: int, save_name: str, args: Mapping, eval_file: str, train_file: str) -> None:
        """
        Trains a model on batches of texts, position indices of the target token, and labels (lemma annotation) for the target token.

        Args:
            num_epochs (int): Number of training epochs
            save_name (str): Path to file where trained model should be saved.
            eval_file (str): Path to the dev set file for evaluating model checkpoints each epoch.
            train_file (str): Path to data file, containing tokenized text sentences, token index and true label for token lemma on each line.
        """
        # Put model on GPU (if possible)
        device = default_device()

        if not train_file:
            raise ValueError("Cannot train model - no train_file supplied!")

        dataset = utils.Dataset(train_file, get_counts=self.weighted_loss, batch_size=args.get("batch_size", DEFAULT_BATCH_SIZE))
        label_decoder = dataset.label_decoder
        upos_to_id = dataset.upos_to_id
        self.output_dim = len(label_decoder)
        logger.info(f"Loaded dataset successfully from {train_file}")
        logger.info(f"Using label decoder: {label_decoder}  Output dimension: {self.output_dim}")
        logger.info(f"Target words: {dataset.target_words}")

        self.model = self.build_model(label_decoder, upos_to_id, dataset.known_words, dataset.target_words, set(dataset.target_upos))
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.to(device)
        logger.info(f"Training model on device: {device}. {next(self.model.parameters()).device}")

        if os.path.exists(save_name) and not args.get('force', False):
            raise FileExistsError(f"Save name {save_name} already exists; training would overwrite previous file contents. Aborting...")

        if self.weighted_loss:
            self.configure_weighted_loss(label_decoder, dataset.counts)

        # Put the criterion on GPU too
        logger.debug(f"Criterion on {next(self.model.parameters()).device}")
        self.criterion = self.criterion.to(next(self.model.parameters()).device)

        best_model, best_f1 = None, float("-inf")  # Used for saving checkpoints of the model
        for epoch in range(num_epochs):
            # go over entire dataset with each epoch
            for sentences, positions, upos_tags, labels in tqdm(dataset):
                assert len(sentences) == len(positions) == len(labels), f"Input sentences, positions, and labels are of unequal length ({len(sentences), len(positions), len(labels)})"

                self.optimizer.zero_grad()
                outputs = self.model(positions, sentences, upos_tags)

                # Compute loss, which is different if using CE or BCEWithLogitsLoss
                if self.weighted_loss:  # BCEWithLogitsLoss requires a vector for target where probability is 1 on the true label class, and 0 on others.
                    # TODO: three classes?
                    targets = torch.stack([torch.tensor([1, 0]) if label == 0 else torch.tensor([0, 1]) for label in labels]).to(dtype=torch.float32).to(device)
                    # should be shape size (batch_size, 2)
                else:  # CELoss accepts target as just raw label
                    targets = labels.to(device)

                loss = self.criterion(outputs, targets)

                loss.backward()
                self.optimizer.step()

            logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")
            if eval_file:
                # Evaluate model on dev set to see if it should be saved.
                _, _, _, f1 = evaluate_model(self.model, eval_file, is_training=True)
                logger.info(f"Weighted f1 for model: {f1}")
                if f1 > best_f1:
                    best_f1 = f1
                    self.model.save(save_name)
                    logger.info(f"New best model: weighted f1 score of {f1}.")
            else:
                self.model.save(save_name)

