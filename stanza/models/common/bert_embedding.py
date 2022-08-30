import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence, PackedSequence

logger = logging.getLogger('stanza')

BERT_ARGS = {
    "vinai/phobert-base": { "use_fast": True },
    "vinai/phobert-large": { "use_fast": True },
}

def update_max_length(model_name, tokenizer):
    if model_name == 'google/muril-base-cased' or model_name == 'airesearch/wangchanberta-base-att-spm-uncased':
        tokenizer.model_max_length = 512

def load_tokenizer(model_name):
    if model_name:
        # note that use_fast is the default
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("Please install transformers library for BERT support! Try `pip install transformers`.")
        bert_args = BERT_ARGS.get(model_name, dict())
        if not model_name.startswith("vinai/phobert"):
            bert_args["add_prefix_space"] = True
        bert_tokenizer = AutoTokenizer.from_pretrained(model_name, **bert_args)
        update_max_length(model_name, bert_tokenizer)
        return bert_tokenizer
    return None

def load_bert(model_name):
    if model_name:
        # such as: "vinai/phobert-base"
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError("Please install transformers library for BERT support! Try `pip install transformers`.")
        bert_model = AutoModel.from_pretrained(model_name)
        bert_tokenizer = load_tokenizer(model_name)
        return bert_model, bert_tokenizer
    return None, None

def tokenize_manual(model_name, sent, tokenizer):
    """
    Tokenize a sentence manually, using for checking long sentences and PHOBert.
    """
    #replace \xa0 or whatever the space character is by _ since PhoBERT expects _ between syllables
    tokenized = [word.replace("\xa0","_").replace(" ", "_") for word in sent] if model_name.startswith("vinai/phobert") else [word.replace("\xa0"," ") for word in sent]

    #concatenate to a sentence
    sentence = ' '.join(tokenized)

    #tokenize using AutoTokenizer PhoBERT
    tokenized = tokenizer.tokenize(sentence)

    #convert tokens to ids
    sent_ids = tokenizer.convert_tokens_to_ids(tokenized)

    #add start and end tokens to sent_ids
    tokenized_sent = [tokenizer.bos_token_id] + sent_ids + [tokenizer.eos_token_id]

    return tokenized, tokenized_sent

def filter_data(model_name, data, tokenizer = None):
    """
    Filter out the (NER) data that is too long for BERT model.
    """
    if tokenizer is None:
        tokenizer = load_tokenizer(model_name) 
    filtered_data = []
    #eliminate all the sentences that are too long for bert model
    for sent in data:
        sentence = [word[0] for word in sent]
        _, tokenized_sent = tokenize_manual(model_name, sentence, tokenizer)
        
        if len(tokenized_sent) > tokenizer.model_max_length - 2:
            continue

        filtered_data.append(sent)

    logger.info("Eliminated {} datapoints because their length is over maximum size of BERT model. ".format(len(data)-len(filtered_data)))
    
    return filtered_data


def extract_phobert_embeddings(model_name, tokenizer, model, data, device, keep_endpoints):
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

        tokenized, tokenized_sent = tokenize_manual(model_name, sent, tokenizer)

        #add tokenized to list_tokenzied for later checking
        list_tokenized.append(tokenized)

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
            features += feature.clone().detach()

    assert len(features)==size
    assert len(features)==len(processed)

    #process the output
    #only take the vector of the last word piece of a word/ you can do other methods such as first word piece or averaging.
    # idx2+1 compensates for the start token at the start of a sentence
    offsets = [[idx2+1 for idx2, _ in enumerate(list_tokenized[idx]) if (idx2 > 0 and not list_tokenized[idx][idx2-1].endswith("@@")) or (idx2==0)]
                for idx, sent in enumerate(processed)]
    if keep_endpoints:
        # [0] and [-1] grab the start and end representations as well
        offsets = [[0] + off + [-1] for off in offsets]
    processed = [feature[offset] for feature, offset in zip(features, offsets)]

    # This is a list of ltensors
    # Each tensor holds the representation of a sentence extracted from phobert
    return processed

BAD_TOKENIZERS = ('bert-base-german-cased', 'dbmdz/bert-base-german-cased', 'google/muril-base-cased', 'l3cube-pune/marathi-roberta')

def fix_german_tokens(tokenizer, data):
    """
    Patch bert tokenizers with missing characters

    There is an issue that some tokenizers (so far the German ones identified above)
    tokenize soft hyphens or other unknown characters into nothing
    If an entire word is tokenized as a soft hyphen, this means the tokenizer
    simply vaporizes that word.  The result is we're missing an embedding for
    an entire word we wanted to use.

    The solution we take here is to look for any words which get vaporized
    in such a manner, eg `len(token) == 2`, and replace it with a regular "-"
    """
    new_data = []
    for sentence in data:
        tokenized = tokenizer(sentence, is_split_into_words=False).input_ids
        new_sentence = [word if len(token) > 2 else "-" for word, token in zip(sentence, tokenized)]
        new_data.append(new_sentence)
    return new_data

def extract_xlnet_embeddings(model_name, tokenizer, model, data, device, keep_endpoints):
    # using attention masks makes contextual embeddings much more useful for downstream tasks
    tokenized = tokenizer(data, is_split_into_words=True, return_offsets_mapping=False, return_attention_mask=False)
    #tokenized = tokenizer(data, padding="longest", is_split_into_words=True, return_offsets_mapping=False, return_attention_mask=True)

    list_offsets = [[None] * (len(sentence)+2) for sentence in data]
    for idx in range(len(data)):
        offsets = tokenized.word_ids(batch_index=idx)
        list_offsets[idx][0] = 0
        for pos, offset in enumerate(offsets):
            if offset is None:
                break
            # this uses the last token piece for any offset by overwriting the previous value
            # this will be one token earlier
            # we will add a <pad> to the start of each sentence for the endpoints
            list_offsets[idx][offset+1] = pos + 1
        list_offsets[idx][-1] = list_offsets[idx][-2] + 1
        if any(x is None for x in list_offsets[idx]):
            raise ValueError("OOPS, hit None when preparing to use Bert\ndata[idx]: {}\noffsets: {}\nlist_offsets[idx]: {}".format(data[idx], offsets, list_offsets[idx], tokenized))

        if len(offsets) > tokenizer.model_max_length - 2:
            logger.error("Invalid size, max size: %d, got %d %s", tokenizer.model_max_length, len(offsets), data[idx])
            raise TextTooLongError(len(offsets), tokenizer.model_max_length, idx, " ".join(data[idx]))

    features = []
    for i in range(int(math.ceil(len(data)/128))):
        with torch.no_grad():
            # TODO: find a suitable representation for attention masks for xlnet
            # xlnet base on WSJ:
            # sep_token_id at beginning, cls_token_id at end:     0.9441
            # bos_token_id at beginning, eos_token_id at end:     0.9463
            # bos_token_id at beginning, sep_token_id at end:     0.9459
            # bos_token_id at beginning, cls_token_id at end:     0.9457
            # bos_token_id at beginning, sep/cls at end:          0.9454
            # use the xlnet tokenization with words at end,
            # begin token is last pad, end token is sep, no mask: 0.9463
            # same, but with masks:                               0.9440
            input_ids = [[tokenizer.bos_token_id] + x[:-2] + [tokenizer.eos_token_id] for x in tokenized['input_ids'][128*i:128*i+128]]
            max_len = max(len(x) for x in input_ids)
            attention_mask = torch.zeros(len(input_ids), max_len, dtype=torch.long, device=device)
            for idx, input_row in enumerate(input_ids):
                attention_mask[idx, :len(input_row)] = 1
                if len(input_row) < max_len:
                    input_row.extend([tokenizer.pad_token_id] * (max_len - len(input_row)))
            id_tensor = torch.tensor(input_ids, device=device)
            feature = model(id_tensor, attention_mask=attention_mask, output_hidden_states=True)
            # feature[2] is the same for bert, but it didn't work for
            # older versions of transformers for xlnet
            # feature = feature[2]
            feature = feature.hidden_states
            feature = torch.stack(feature[-4:-1], axis=3).sum(axis=3) / 4
            features += feature.clone().detach()

    processed = []
    #process the output
    if not keep_endpoints:
        #remove the bos and eos tokens
        list_offsets = [sent[1:-1] for sent in list_offsets]
    for feature, offsets in zip(features, list_offsets):
        new_sent = feature[offsets]
        processed.append(new_sent)

    return processed


def extract_bert_embeddings(model_name, tokenizer, model, data, device, keep_endpoints):
    """
    Extract transformer embeddings using a generic roberta extraction
    data: list of list of string (the text tokens)
    """
    if model_name.startswith("vinai/phobert"):
        return extract_phobert_embeddings(model_name, tokenizer, model, data, device, keep_endpoints)

    if isinstance(data, tuple):
        data = list(data)

    if model_name.startswith("xlnet"):
        return extract_xlnet_embeddings(model_name, tokenizer, model, data, device, keep_endpoints)

    if model_name in BAD_TOKENIZERS:
        data = fix_german_tokens(tokenizer, data)

    #add add_prefix_space = True for RoBerTa-- error if not
    # using attention masks makes contextual embeddings much more useful for downstream tasks
    tokenized = tokenizer(data, padding="longest", is_split_into_words=True, return_offsets_mapping=False, return_attention_mask=True)
    list_offsets = [[None] * (len(sentence)+2) for sentence in data]
    for idx in range(len(data)):
        offsets = tokenized.word_ids(batch_index=idx)
        for pos, offset in enumerate(offsets):
            if offset is None:
                continue
            # this uses the last token piece for any offset by overwriting the previous value
            list_offsets[idx][offset+1] = pos
        list_offsets[idx][0] = 0
        list_offsets[idx][-1] = list_offsets[idx][-2] + 1
        #print(list_offsets[idx])
        if any(x is None for x in list_offsets[idx]):
            raise ValueError("OOPS, hit None when preparing to use Bert\ndata[idx]: {}\noffsets: {}\nlist_offsets[idx]: {}".format(data[idx], offsets, list_offsets[idx], tokenized))

        if len(offsets) > tokenizer.model_max_length - 2:
            logger.error("Invalid size, max size: %d, got %d %s", tokenizer.model_max_length, len(offsets), data[idx])
            raise TextTooLongError(len(offsets), tokenizer.model_max_length, idx, " ".join(data[idx]))

    features = []
    for i in range(int(math.ceil(len(data)/128))):
        with torch.no_grad():
            attention_mask = torch.tensor(tokenized['attention_mask'][128*i:128*i+128], device=device)
            id_tensor = torch.tensor(tokenized['input_ids'][128*i:128*i+128], device=device)
            feature = model(id_tensor, attention_mask=attention_mask, output_hidden_states=True)
            # feature[2] is the same for bert, but it didn't work for
            # older versions of transformers for xlnet
            # feature = feature[2]
            feature = feature.hidden_states
            feature = torch.stack(feature[-4:-1], axis=3).sum(axis=3) / 4
            features += feature.clone().detach()

    processed = []
    #process the output
    if not keep_endpoints:
        #remove the bos and eos tokens
        list_offsets = [sent[1:-1] for sent in list_offsets]
    for feature, offsets in zip(features, list_offsets):
        new_sent = feature[offsets]
        processed.append(new_sent)

    return processed
