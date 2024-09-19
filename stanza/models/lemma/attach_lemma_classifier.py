import argparse

from stanza.models.lemma.trainer import Trainer
from stanza.models.lemma_classifier.base_model import LemmaClassifier

def attach_classifier(input_filename, output_filename, classifiers):
    trainer = Trainer(model_file=input_filename)

    for classifier in classifiers:
        classifier = LemmaClassifier.load(classifier)
        trainer.contextual_lemmatizers.append(classifier)

    trainer.save(output_filename)

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Which lemmatizer to start from')
    parser.add_argument('--output', type=str, required=True, help='Where to save the lemmatizer')
    parser.add_argument('--classifier', type=str, required=True, nargs='+', help='Lemma classifier to attach')
    args = parser.parse_args(args)

    attach_classifier(args.input, args.output, args.classifier)

if __name__ == '__main__':
    main()
