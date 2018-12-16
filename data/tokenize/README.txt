This directory stores data used for creating the tokenizer.

Example:

# raw input text
en_ewt.train.txt

# gold training data
en_ewt.train.gold.conllu

# extra train files used during training process
en_ewt-ud-train-mwt.json
en_ewt-ud-train.toklabels

# gold dev data
en_ewt.dev.gold.conllu

# extra dev files used during training process
en_ewt-ud-dev-mwt.json
en_ewt-ud-dev.toklabels

# tokenizer predicted dev data (by model in saved_models/en_ewt_tokenizer.pt)
en_ewt.dev.pred.conllu
