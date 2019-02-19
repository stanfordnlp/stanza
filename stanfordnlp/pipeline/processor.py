"""
base classes for processors
"""

from abc import ABC, abstractmethod


# base class for all processors
class Processor(ABC):

    @abstractmethod
    def process(self, doc):
        pass


# base class for UD processors
class UDProcessor(Processor):

    @abstractmethod
    def process(self, doc):
        pass

    def build_final_config(self, config):
        # set configurations from loaded model
        if self.trainer is not None:
            loaded_args, self.vocab = self.trainer.args, self.trainer.vocab
            # filter out unneeded args from model
            loaded_args = {k: v for k, v in loaded_args.items() if not UDProcessor.filter_out_option(k)}
        else:
            loaded_args = {}
        loaded_args.update(config)
        self.config = loaded_args

    @staticmethod
    def filter_out_option(option):
        options_to_filter = ['cpu', 'cuda', 'dev_conll_gold', 'epochs', 'lang', 'mode', 'save_name', 'shorthand']
        if option.endswith('_file') or option.endswith('_dir'):
            return True
        elif option in options_to_filter:
            return True
        else:
            return False






