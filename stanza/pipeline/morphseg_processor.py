from stanza.pipeline.processor import UDProcessor, register_processor
from stanza.pipeline._constants import MORPHSEG, TOKENIZE


@register_processor(name=MORPHSEG)
class MorphSegProcessor(UDProcessor):
    PROVIDES_DEFAULT = {MORPHSEG}
    REQUIRES_DEFAULT = {TOKENIZE}

    def __init__(self, config, pipeline, device):
        self._config = config
        self._pipeline = pipeline
        self._set_up_requires()
        self._set_up_provides()
        self._set_up_model(config, pipeline, device)

    def _set_up_model(self, config, pipeline, device):
        try:
            from morphseg import MorphemeSegmenter
        except ImportError:
            raise ImportError(
                "morphseg is required for morpheme segmentation. "
                "Install it with: pip install morphseg"
            )

        lang = config.get('lang', 'en')
        model_path = config.get('morphseg_model_path', None)

        if model_path:
            self._segmenter = MorphemeSegmenter(
                lang=lang,
                load_pretrained=False,
                model_filepath=model_path,
                is_local=True
            )
        else:
            self._segmenter = MorphemeSegmenter(
                lang=lang,
                load_pretrained=True
            )

    def process(self, document):
        # Collect all words from all sentences
        all_words = []
        word_mapping = []  # Track which sentence and word index each prediction belongs to

        for sent_idx, sent in enumerate(document.sentences):
            if not sent.words:
                continue
            for word_idx, word in enumerate(sent.words):
                all_words.append(word.text)
                word_mapping.append((sent_idx, word_idx))

        if not all_words:
            return document

        # Prepare input for morphseg (it expects normalized, lowercased character lists)
        word_char_lists = [
            list(self._segmenter.normalize_for_morphology(word))
            for word in all_words
        ]

        # Batch predict using the internal sequence_labeller
        predictions = self._segmenter.sequence_labeller.predict(sources=word_char_lists)

        # Extract segmentations from predictions
        from morphseg.training.oracle import rules2sent
        segmentations = [
            rules2sent(
                source=[align_pos.symbol for align_pos in pred.alignment],
                actions=pred.prediction
            ).split(' @@')  # Split by morphseg's default delimiter
            for pred in predictions
        ]

        # Assign segmentations back to words
        for (sent_idx, word_idx), seg in zip(word_mapping, segmentations):
            document.sentences[sent_idx].words[word_idx].morphemes = seg

        return document
