import stanza

from stanza.models.common import doc
from stanza.models.tokenization.data import TokenizationDataset
from stanza.models.tokenization.utils import predict, decode_predictions

def resplit_mwt(tokens, pipeline, keep_tokens=True):
    """
    Uses the tokenize processor and the mwt processor in the pipeline to resplit tokens into MWT

    tokens: a list of list of string
    pipeline: a Stanza pipeline which contains, at a minimum, tokenize and mwt

    keep_tokens: if True, enforce the old token boundaries by modify
      the results of the tokenize inference.
      Otherwise, use whatever new boundaries the model comes up with.

    between running the tokenize model and breaking the text into tokens,
    we can update all_preds to use the original token boundaries
    (if and only if keep_tokens == True)

    This method returns a Document with just the tokens and words annotated.
    """
    if "tokenize" not in pipeline.processors:
        raise ValueError("Need a Pipeline with a valid tokenize processor")
    if "mwt" not in pipeline.processors:
        raise ValueError("Need a Pipeline with a valid mwt processor")
    tokenize_processor = pipeline.processors["tokenize"]
    mwt_processor = pipeline.processors["mwt"]
    fake_text = "\n\n".join(" ".join(sentence) for sentence in tokens)

    # set up batches
    batches = TokenizationDataset(tokenize_processor.config,
                                  input_text=fake_text,
                                  vocab=tokenize_processor.vocab,
                                  evaluation=True,
                                  dictionary=tokenize_processor.trainer.dictionary)

    all_preds, all_raw = predict(trainer=tokenize_processor.trainer,
                                 data_generator=batches,
                                 batch_size=tokenize_processor.trainer.args['batch_size'],
                                 max_seqlen=tokenize_processor.config.get('max_seqlen', tokenize_processor.MAX_SEQ_LENGTH_DEFAULT),
                                 use_regex_tokens=True,
                                 num_workers=tokenize_processor.config.get('num_workers', 0))

    if keep_tokens:
        for sentence, pred in zip(tokens, all_preds):
            char_idx = 0
            for word in sentence:
                if len(word) > 0:
                    pred[char_idx:char_idx+len(word)-1] = 0
                if pred[char_idx+len(word)-1] == 0:
                    pred[char_idx+len(word)-1] = 1
                char_idx += len(word) + 1

    _, _, document = decode_predictions(vocab=tokenize_processor.vocab,
                                        mwt_dict=None,
                                        orig_text=fake_text,
                                        all_raw=all_raw,
                                        all_preds=all_preds,
                                        no_ssplit=True,
                                        skip_newline=tokenize_processor.trainer.args['skip_newline'],
                                        use_la_ittb_shorthand=tokenize_processor.trainer.args['shorthand'] == 'la_ittb')

    document = doc.Document(document, fake_text)
    mwt_processor.process(document)
    return document

def main():
    pipe = stanza.Pipeline("en", processors="tokenize,mwt", package="gum")
    tokens = [["I", "can't", "believe", "it"], ["I can't", "sleep"]]
    doc = resplit_mwt(tokens, pipe)
    print(doc)

    doc = resplit_mwt(tokens, pipe, keep_tokens=False)
    print(doc)

if __name__ == '__main__':
    main()
