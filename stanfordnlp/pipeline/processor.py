"""
Base classes for processors
"""

from abc import ABC, abstractmethod


class ProcessorRequirementsException(Exception):
    """
    Exception indicating a processor's requirements will not be met
    """

    def __init__(self, processors_list, err_processor, provided_reqs):
        self._processor_type = type(err_processor).__name__
        self._processors_list = processors_list
        self._provided_reqs = provided_reqs
        self._requires = err_processor.requires
        super().__init__(self.build_message())

    @property
    def processor_type(self):
        return self._processor_type

    @property
    def processors_list(self):
        return self._processors_list

    @property
    def provided_reqs(self):
        return self._provided_reqs

    @property
    def requires(self):
        return self._requires
    
    def build_message(self):
        return f"""
        ---
        Pipeline Requirements Error!
        \tProcessor: {self.processor_type}
        \tPipeline processors list: {','.join(self.processors_list)}
        \tProcessor Requirements: {self.requires)}
        \t\t- fulfilled: {self.requires.intersection(self.provided_reqs)}
        \t\t- missing: {self.requires - self.provided_reqs}
        
        The processors list provided for this pipeline is invalid.  Please make sure all prerequisites are met for
        every processor.
        """.lstrip()


class Processor(ABC):
    """
    Base class for all processors
    """

    @abstractmethod
    def process(self, doc):
        """
        Process a Document.  This is the main method of a processor.
        :param doc: the input Document
        :return: None
        """
        pass

    def _set_provides(self):
        """
        Announce what processor requirements this processor fulfills.  By default this is a hard coded list.
        Subclasses may fulfill different requirements based on configurations.
        :return: None
        """
        self._provides = self.__class__.PROVIDES_DEFAULT

    def _set_requires(self):
        """
        Set up the requirements for this processor.  By default this is a hard coded list.  Subclasses may have
        different requirements based on configurations and should override this method.
        :return: None
        """
        self._requires = self.__class__.REQUIRES_DEFAULT

    @property
    def pipeline(self):
        return self._pipeline

    @property
    def provides(self):
        return self._provides

    @property
    def requires(self):
        return self._requires

    def _check_requirements(self):
        """
        Given a list of fulfilled requirements, check if all of this processor's requirements are met or not.
        If not, raise an Exception.
        :return: None
        """
        provided_reqs = set.union(*[processor.provides for processor in self.pipeline.loaded_processors])
        if self.requires - provided_reqs:
            raise ProcessorRequirementsException(self.pipeline.processor_names, self, provided_reqs)


class UDProcessor(Processor):
    """
    Base class for the neural UD Processors (tokenize,mwt,pos,lemma,depparse)
    """

    @abstractmethod
    def process(self, doc):
        pass

    def _build_final_config(self, config):
        """
        Finalize the configurations for this processor, based off of values from a UD model.
        :param config: config for the processor
        :return: None
        """
        # set configurations from loaded model
        if self._trainer is not None:
            loaded_args, self._vocab = self._trainer.args, self._trainer.vocab
            # filter out unneeded args from model
            loaded_args = {k: v for k, v in loaded_args.items() if not UDProcessor.filter_out_option(k)}
        else:
            loaded_args = {}
        loaded_args.update(config)
        self._config = loaded_args

    @staticmethod
    def filter_out_option(option):
        """
        Filter out non-processor configurations
        :param option: string to potentially filter out
        :return: bool representing whether or not this keyword should be filtered out
        """
        options_to_filter = ['cpu', 'cuda', 'dev_conll_gold', 'epochs', 'lang', 'mode', 'save_name', 'shorthand']
        if option.endswith('_file') or option.endswith('_dir'):
            return True
        elif option in options_to_filter:
            return True
        else:
            return False






