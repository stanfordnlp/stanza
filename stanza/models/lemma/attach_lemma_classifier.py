"""
Attaches an existing LemmaClassifier (or potentially any contextual lemmatizer) to an existing lemma.trainer.Trainer

Used by the run_lemma.py script to build a combined lemmatizer w/ contextual lemmatizers.

Example usage for assembling an English lemmatizer.  Note that you can list multiple classifiers at once:

python3 stanza/models/lemma/attach_lemma_classifier.py
    --input saved_models/lemma/en_combined_charlm_lemmatizer.pt
    --output en_combined_charlm.pt
    --remove_existing
    --classifier saved_models/lemma_classifier/en_combined.s_lemma_classifier.pt
                 saved_models/lemma_classifier/en_combined.her_lemma_classifier.pt
"""

import argparse
import logging

from stanza.models.lemma.trainer import Trainer
from stanza.models.lemma_classifier.base_model import LemmaClassifier

logger = logging.getLogger('stanza')

def attach_classifier(input_filename, output_filename, classifiers, remove_existing):
    logger.info("Attaching lemmatizers to %s", input_filename)
    trainer = Trainer(model_file=input_filename)

    if remove_existing:
        trainer.contextual_lemmatizers = []

    for classifier in classifiers:
        logger.info("Loading %s", classifier)
        classifier = LemmaClassifier.load(classifier)
        logger.info("  Lemma classifier operates on word(s) %s with tag %s", classifier.target_words, classifier.target_upos)
        trainer.contextual_lemmatizers.append(classifier)

    logger.info("Total contextual_lemmatizers: %d", len(trainer.contextual_lemmatizers))
    trainer.save(output_filename)

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Which lemmatizer to start from')
    parser.add_argument('--output', type=str, required=True, help='Where to save the lemmatizer')
    parser.add_argument('--classifier', type=str, required=True, nargs='+', help='Lemma classifier to attach')
    parser.add_argument('--remove_existing', default=False, action='store_true', help='Remove any existing lemma classifiers')
    args = parser.parse_args(args)

    attach_classifier(args.input, args.output, args.classifier, args.remove_existing)

if __name__ == '__main__':
    main()
