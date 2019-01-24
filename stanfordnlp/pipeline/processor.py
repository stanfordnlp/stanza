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
        loaded_args, self.vocab = self.trainer.args, self.trainer.vocab
        # filter out unneeded args from model
        loaded_args = {k: v for k, v in loaded_args.items() if k in self.model_options}
        # overwrite with config for processor
        loaded_args.update(config)
        self.config = loaded_args




