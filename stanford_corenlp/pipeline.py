"""
Module for NLP processing pipeline
"""

from stanford_corenlp.algorithms.ssplit import SentenceSplitterAnnotator
from stanford_corenlp.algorithms.tokenize import TokenizerAnnotator

# by default run all annotators
DEFAULT_ANNOTATORS_LIST = "tokenize,ssplit,pos,ner,depparse"

# default settings is empty dictionary
# Example: {"tokenize.whitespace": True, "ssplit.newline_only": True}
DEFAULT_PIPELINE_SETTINGS = {}

# map names to annotators
NAMES_TO_ANNOTATORS = {
    "ssplit": SentenceSplitterAnnotator,
    "tokenize": TokenizerAnnotator
}

class Pipeline:
    """Class for running an NLP pipeline on documents"""

    def __init__(self, annotators=DEFAULT_ANNOTATORS_LIST, settings=DEFAULT_PIPELINE_SETTINGS):
        """Initialize the pipeline with a comma separated list of annotators

            Setting up the pipeline:

            1. Verify pipeline validity (each annotator has its prerequisites met by an earlier annotator)
            2. Build annotators according to settings, cache resources

        """
        # set up annotators list and settings
        self._annotator_names = annotators.split(",")
        self._annotators = []
        self._settings = settings
        # check if pipeline is valid, if not raise exception
        valid_pipeline, prob_annotator, missing_prereqs = self.validate_pipeline()
        if not valid_pipeline:
            raise RuntimeError(
                "Pipeline as configured is not valid!" +
                "\nLacking prerequisite annotators for: "+prob_annotator +
                "\nMissing prerequisites: "+missing_prereqs)
        # if pipeline is valid, iterate through each annotator name and build an annotator
        self.build_annotators()

    @classmethod
    def validate_pipeline(cls):
        """Validate that each annotator has its prerequisites"""
        return True, None, None

    def build_annotators(self):
        """Build each annotator in the specified annotator list and add to list of annotators"""
        for annotator_name in self.annotator_names:
            curr_annotator = NAMES_TO_ANNOTATORS[annotator_name](self.settings)
            self.add_annotator(curr_annotator)

    @property
    def annotators(self):
        """Access the list of annotators for this pipeline"""
        return self._annotators

    @property
    def annotator_names(self):
        """Access the list of annotators for this pipeline"""
        return self._annotator_names

    @property
    def settings(self):
        """Access the list of annotators for this pipeline"""
        return self._settings

    def add_annotator(self, annotator):
        self._annotators.append(annotator)

    def annotate(self, doc):
        """Method which runs each annotator on the document"""
        for annotator in self.annotators:
            annotator.annotate(doc)

    def add_annotator(self, annotator):
        self._annotators.append(annotator)

