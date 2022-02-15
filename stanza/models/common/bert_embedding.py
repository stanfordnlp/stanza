import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence, PackedSequence
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger('stanza')

BERT_ARGS = {
    "vinai/phobert-base": { "use_fast": True },
    "vinai/phobert-large": { "use_fast": True },
}


def load_tokenizer(model_name):
    if model_name:
        # note that use_fast is the default
        bert_args = BERT_ARGS.get(model_name, dict())
        if not model_name.startswith("vinai/phobert"):
            bert_args["add_prefix_space"] = True
        bert_tokenizer = AutoTokenizer.from_pretrained(model_name, **bert_args)
        return bert_tokenizer
    return None

def load_bert(model_name):
    if model_name:
        # such as: "vinai/phobert-base"
        bert_model = AutoModel.from_pretrained(model_name)
        bert_tokenizer = load_tokenizer(model_name)
        return bert_model, bert_tokenizer
    return None, None

def extract_phobert_embeddings(tokenizer, model, data, device):
    """
    Extract transformer embeddings using a method specifically for phobert
    Since phobert doesn't have the is_split_into_words / tokenized.word_ids(batch_index=0)
    capability, we instead look for @@ to denote a continued token.
    data: list of list of string (the text tokens)
    """
    processed = [] # final product, returns the list of list of word representation
    tokenized_sents = [] # list of sentences, each is a torch tensor with start and end token
    list_tokenized = [] # list of tokenized sentences from phobert
    for idx, sent in enumerate(data):
        #replace \xa0 or whatever the space character is by _ since PhoBERT expects _ between syllables
        tokenized = [word.replace("\xa0","_").replace(" ", "_") for word in sent]

        #concatenate to a sentence
        sentence = ' '.join(tokenized)

        #tokenize using AutoTokenizer PhoBERT
        tokenized = tokenizer.tokenize(sentence)

        #add tokenized to list_tokenzied for later checking
        list_tokenized.append(tokenized)

        #convert tokens to ids
        sent_ids = tokenizer.convert_tokens_to_ids(tokenized)

        #add start and end tokens to sent_ids
        tokenized_sent = [tokenizer.bos_token_id] + sent_ids + [tokenizer.eos_token_id]

        if len(tokenized_sent) > tokenizer.model_max_length:
            logger.error("Invalid size, max size: %d, got %d %s", tokenizer.model_max_length, len(tokenized_sent), data[idx])
            #raise TextTooLongError(len(tokenized_sent), tokenizer.model_max_length, idx, " ".join(data[idx]))

        #add to tokenized_sents
        tokenized_sents.append(torch.tensor(tokenized_sent).detach())

        processed_sent = []
        processed.append(processed_sent)

        # done loading bert emb

    size = len(tokenized_sents)

    #padding the inputs
    tokenized_sents_padded = torch.nn.utils.rnn.pad_sequence(tokenized_sents,batch_first=True,padding_value=tokenizer.pad_token_id)

    features = []

    # Feed into PhoBERT 128 at a time in a batch fashion. In testing, the loop was
    # run only 1 time as the batch size seems to be 30
    for i in range(int(math.ceil(size/128))):
        with torch.no_grad():
            feature = model(tokenized_sents_padded[128*i:128*i+128].clone().detach().to(device), output_hidden_states=True)
            # averaging the last four layers worked well for non-VI languages
            feature = feature[2]
            feature = torch.stack(feature[-4:-1], axis=3).sum(axis=3) / 4
            features += feature.clone().detach().cpu()

    assert len(features)==size
    assert len(features)==len(processed)

    #process the output
    #only take the vector of the last word piece of a word/ you can do other methods such as first word piece or averaging.
    # idx2+1 compensates for the start token at the start of a sentence
    # [0] and [-1] grab the start and end representations as well
    offsets = [[idx2+1 for idx2, _ in enumerate(list_tokenized[idx]) if (idx2 > 0 and not list_tokenized[idx][idx2-1].endswith("@@")) or (idx2==0)] 
                for idx, sent in enumerate(processed)]
    processed = [feature[offset] for feature, offset in zip(features, offsets)]

    # This is a list of ltensors
    # Each tensor holds the representation of a sentence extracted from phobert
    return processed

def extract_bert_embeddings(model_name, tokenizer, model, data, device):
    """
    Extract transformer embeddings using a generic roberta extraction
    data: list of list of string (the text tokens)
    """
    if model_name.startswith("vinai/phobert"):
        return extract_phobert_embeddings(tokenizer, model, data, device)

    #add add_prefix_space = True for RoBerTa-- error if not
    tokenized = tokenizer(data, padding="longest", is_split_into_words=True, return_offsets_mapping=False, return_attention_mask=False)
    list_offsets = [[None] * (len(sentence)+2) for sentence in data]
    for idx in range(len(data)):
        offsets = tokenized.word_ids(batch_index=idx)
        for pos, offset in enumerate(offsets):
            if offset is None:
                continue
            # this uses the last token piece for any offset by overwriting the previous value
            list_offsets[idx][offset+1] = pos
        list_offsets[idx][0] = 0
        list_offsets[idx][-1] = -1

        if len(offsets) > tokenizer.model_max_length:
            logger.error("Invalid size, max size: %d, got %d %s", tokenizer.model_max_length, len(offsets), data[idx])
            raise TextTooLongError(len(offsets), tokenizer.model_max_length, idx, " ".join(data[idx]))

    features = []
    for i in range(int(math.ceil(len(data)/128))):
        with torch.no_grad():
            feature = model(torch.tensor(tokenized['input_ids'][128*i:128*i+128]).to(device), output_hidden_states=True)
            feature = feature[2]
            feature = torch.stack(feature[-4:-1], axis=3).sum(axis=3) / 4
            features += feature.clone().detach().cpu()

    processed = []
    #remove the bos and eos tokens
    list_offsets = [ sent[1:-1] for sent in list_offsets]
    #process the output
    for feature, offsets in zip(features, list_offsets):
        new_sent = feature[offsets]
        processed.append(new_sent)

    return processed