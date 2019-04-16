"""
Base classes for processors
"""

from abc import ABC, abstractmethod


class ProcessorRequirementsException(Exception):
    """ Exception indicating a processor's requirements will not be met """

    def __init__(self, processors_list, err_processor, provided_reqs):
        self._err_processor = err_processor
        # mark the broken processor as inactive, drop resources
        self.err_processor.mark_inactive()
        self._processors_list = processors_list
        self._provided_reqs = provided_reqs
        self.build_message()

    @property
    def err_processor(self):
        """ The processor that raised the exception """
        return self._err_processor

    @property
    def processor_type(self):
        return type(self.err_processor).__name__

    @property
    def processors_list(self):
        return self._processors_list

    @property
    def provided_reqs(self):
        return self._provided_reqs

    def build_message(self):
        self.message = (f"---\nPipeline Requirements Error!\n"
                        f"\tProcessor: {self.processor_type}\n"
                        f"\tPipeline processors list: {','.join(self.processors_list)}\n"
                        f"\tProcessor Requirements: {self.err_processor.requires}\n"
                        f"\t\t- fulfilled: {self.err_processor.requires.intersection(self.provided_reqs)}\n"
                        f"\t\t- missing: {self.err_processor.requires - self.provided_reqs}\n"
                        f"\nThe processors list provided for this pipeline is invalid.  Please make sure all "
                        f"prerequisites are met for every processor.\n\n")

    def __str__(self):
        return self.message


class Processor(ABC):
    """ Base class for all processors """

    @abstractmethod
    def process(self, doc):
        """ Process a Document.  This is the main method of a processor. """
        pass

    def _set_up_provides(self):
        """ Set up what processor requirements this processor fulfills.  Default is to use a class defined list. """
        self._provides = self.__class__.PROVIDES_DEFAULT

    def _set_up_requires(self):
        """ Set up requirements for this processor.  Default is to use a class defined list. """
        self._requires = self.__class__.REQUIRES_DEFAULT

    @property
    def config(self):
        """ Configurations for the processor """
        return self._config

    @property
    def pipeline(self):
        """ The pipeline that this processor belongs to """
        return self._pipeline

    @property
    def provides(self):
        return self._provides

    @property
    def requires(self):
        return self._requires

    def _check_requirements(self):
        """ Given a list of fulfilled requirements, check if all of this processor's requirements are met or not. """
        provided_reqs = set.union(*[processor.provides for processor in self.pipeline.loaded_processors]+[set([])])
        if self.requires - provided_reqs:
            raise ProcessorRequirementsException(self.pipeline.processor_names, self, provided_reqs)


class UDProcessor(Processor):
    """ Base class for the neural UD Processors (tokenize,mwt,pos,lemma,depparse) """
    def __init__(self, config, pipeline, use_gpu):
        # overall config for the processor
        self._config = None
        # pipeline building this processor (presently processors are only meant to exist in one pipeline)
        self._pipeline = pipeline
        # UD model resources, set up is processor specific
        self._pretrain = None
        self._trainer = None
        self._vocab = None
        self._set_up_model(config, use_gpu)
        # run set up process
        # build the final config for the processor
        self._set_up_final_config(config)
        # set up what annotations are required based on config
        self._set_up_requires()
        # set up what annotations are provided based on config
        self._set_up_provides()
        # given pipeline constructing this processor, check if requirements are met, throw exception if not
        self._check_requirements()

    @abstractmethod
    def _set_up_model(self, config, gpu):
        pass

    def _set_up_final_config(self, config):
        """ Finalize the configurations for this processor, based off of values from a UD model. """
        # set configurations from loaded model
        if self._trainer is not None:
            loaded_args, self._vocab = self._trainer.args, self._trainer.vocab
            # filter out unneeded args from model
            loaded_args = {k: v for k, v in loaded_args.items() if not UDProcessor.filter_out_option(k)}
        else:
            loaded_args = {}
        loaded_args.update(config)
        self._config = loaded_args

    def mark_inactive(self):
        """ Drop memory intensive resources if keeping this processor around for reasons other than running it. """
        self._trainer = None
        self._vocab = None

    @property
    def pretrain(self):
        return self._pretrain

    @property
    def trainer(self):
        return self._trainer

    @property
    def vocab(self):
        return self._vocab

    @staticmethod
    def filter_out_option(option):
        """ Filter out non-processor configurations """
        options_to_filter = ['cpu', 'cuda', 'dev_conll_gold', 'epochs', 'lang', 'mode', 'save_name', 'shorthand']
        if option.endswith('_file') or option.endswith('_dir'):
            return True
        elif option in options_to_filter:
            return True
        else:
            return False






