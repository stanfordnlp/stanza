"""
For a dataset produced by prepare_sentiment_dataset, add constituency parses.

Obviously this will only work on languages that have a constituency parser
"""

import argparse
import os

import stanza
from stanza.models.classifiers.data import read_dataset
from stanza.models.classifiers.utils import WVType
from stanza.models.mwt.utils import resplit_mwt
from stanza.utils.datasets.sentiment import prepare_sentiment_dataset
from stanza.utils.datasets.sentiment import process_utils
import stanza.utils.default_paths as default_paths

SHARDS = ("train", "dev", "test")

def main():
    parser = argparse.ArgumentParser()
    # TODO: allow multiple files?
    parser.add_argument('dataset', type=str, help="Dataset (or a single file) to process")
    parser.add_argument('--output', type=str, help="Write the processed data here instead of clobbering")
    parser.add_argument('--constituency_package', type=str, default=None, help="Constituency model to use for parsing")
    parser.add_argument('--constituency_model', type=str, default=None, help="Specific model file to use for parsing")
    parser.add_argument('--retag_package', type=str, default=None, help="Which tagger to use for retagging")
    parser.add_argument('--split_mwt', action='store_true', help="Split MWT from the original sentences if the language has MWT")
    parser.add_argument('--lang', type=str, default=None, help="Which language the dataset/file is in.  If not specified, will try to use the dataset name")
    args = parser.parse_args()

    if os.path.exists(args.dataset):
        expected_files = [args.dataset]
        if args.output:
            output_files = [args.output]
        else:
            output_files = expected_files
        if not args.lang:
            _, filename = os.path.split(args.dataset)
            args.lang = filename.split("_")[0]
            print("Guessing lang=%s based on the filename %s" % (args.lang, filename))
    else:
        paths = default_paths.get_default_paths()
        # TODO: one of the side effects of the tass2020 dataset is to make a bunch of extra files
        # Perhaps we could have the prepare_sentiment_dataset script return a list of those files
        expected_files = [os.path.join(paths['SENTIMENT_DATA_DIR'], '%s.%s.json' % (args.dataset, shard)) for shard in SHARDS]
        if args.output:
            output_files = [os.path.join(paths['SENTIMENT_DATA_DIR'], '%s.%s.json' % (args.output, shard)) for shard in SHARDS]
        else:
            output_files = expected_files
        for filename in expected_files:
            if not os.path.exists(filename):
                print("Cannot find expected dataset file %s - rebuilding dataset" % filename)
                prepare_sentiment_dataset.main(args.dataset)
                break
        if not args.lang:
            args.lang, _ = args.dataset.split("_", 1)
            print("Guessing lang=%s based on the dataset name" % args.lang)


    pipeline_args = {"lang": args.lang,
                     "processors": "tokenize,pos,constituency",
                     "tokenize_pretokenized": True,
                     "pos_tqdm": True,
                     "constituency_tqdm": True}
    package = {}
    if args.constituency_package is not None:
        package["constituency"] = args.constituency_package
    if args.retag_package is not None:
        package["pos"] = args.retag_package
    if package:
        pipeline_args["package"] = package
    if args.constituency_model is not None:
        pipeline_args["constituency_model_path"] = args.constituency_model
    pipe = stanza.Pipeline(**pipeline_args)

    if args.split_mwt:
        # TODO: allow for different tokenize packages
        mwt_pipe = stanza.Pipeline(lang=args.lang, processors="tokenize")
        if "mwt" in mwt_pipe.processors:
            print("This language has MWT.  Will resplit any MWTs found in the dataset")
        else:
            print("--split_mwt was requested, but %s does not support MWT!" % args.lang)
            args.split_mwt = False

    for filename, output_filename in zip(expected_files, output_files):
        dataset = read_dataset(filename, WVType.OTHER, 1)
        text = [x.text for x in dataset]
        if args.split_mwt:
            print("Resplitting MWT in %d sentences from %s" % (len(dataset), filename))
            doc = resplit_mwt(text, mwt_pipe)
            print("Parsing %d sentences from %s" % (len(dataset), filename))
            doc = pipe(doc)
        else:
            print("Parsing %d sentences from %s" % (len(dataset), filename))
            doc = pipe(text)

        assert len(dataset) == len(doc.sentences)
        for datum, sentence in zip(dataset, doc.sentences):
            datum.constituency = sentence.constituency

        process_utils.write_list(output_filename, dataset)

if __name__ == '__main__':
    main()
