import argparse

import json
import torch
from tqdm import tqdm

from stanza.models.coref.model import CorefModel


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("experiment")
    argparser.add_argument("input_file")
    argparser.add_argument("output_file")
    argparser.add_argument("--config-file", default="config.toml")
    argparser.add_argument("--batch-size", type=int,
                           help="Adjust to override the config value if you're"
                                " experiencing out-of-memory issues")
    argparser.add_argument("--weights",
                           help="Path to file with weights to load."
                                " If not supplied, in the latest"
                                " weights of the experiment will be loaded;"
                                " if there aren't any, an error is raised.")
    args = argparser.parse_args()

    model = CorefModel.load_model(path=args.weights,
                                  map_location="cpu",
                                  ignore={"bert_optimizer", "general_optimizer",
                                          "bert_scheduler", "general_scheduler"})
    if args.batch_size:
        model.config.a_scoring_batch_size = args.batch_size
    model.training = False

    try:
        with open(args.input_file, encoding="utf-8") as fin:
            input_data = json.load(fin)
    except json.decoder.JSONDecodeError:
        # read the old jsonlines format if necessary
        with open(args.input_file, encoding="utf-8") as fin:
            text = "[" + ",\n".join(fin) + "]"
        input_data = json.loads(text)
    docs = [model.build_doc(doc) for doc in input_data]

    with torch.no_grad():
        for doc in tqdm(docs, unit="docs"):
            result = model.run(doc)
            doc["span_clusters"] = result.span_clusters
            doc["word_clusters"] = result.word_clusters

            for key in ("word2subword", "subwords", "word_id", "head2span"):
                del doc[key]

    with open(args.output_file, mode="w") as fout:
        for doc in docs:
            json.dump(doc, fout)
