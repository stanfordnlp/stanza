"""
Base classes for processors
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, ClassVar, Optional, Protocol, TypeVar, Union, runtime_checkable

import torch

from stanza.models.common.doc import Document
from stanza.models.common.pretrain import Pretrain
from stanza.models.common.vocab import BaseMultiVocab, BaseVocab
from stanza.pipeline.registry import NAME_TO_PROCESSOR_CLASS, PIPELINE_NAMES, PROCESSOR_VARIANTS

_ProcessorT = TypeVar("_ProcessorT", bound="Processor")
_ProcessorVariantT = TypeVar("_ProcessorVariantT", bound="ProcessorVariant")
_VariantConfigT = TypeVar("_VariantConfigT")
ProcessorDevice = Union[str, torch.device]
ProcessorVocab = Union[BaseVocab, BaseMultiVocab]


@runtime_checkable
class _VariantRequirementsInitializer(Protocol):
    def _set_up_requires(self) -> None:
        ...


@runtime_checkable
class _VariantRequirements(Protocol):
    _requires: set[str]


@runtime_checkable
class _VariantDefaultRequirements(Protocol):
    REQUIRES_DEFAULT: set[str]


class ProcessorRequirementsException(Exception):
    """ Exception indicating a processor's requirements will not be met """

    def __init__(
            self,
            processors_list: list[str],
            err_processor: Processor,
            provided_reqs: set[str],
        ) -> None:
        self._err_processor = err_processor
        # mark the broken processor as inactive, drop resources
        self.err_processor.mark_inactive()
        self._processors_list = processors_list
        self._provided_reqs = provided_reqs
        self.build_message()

    @property
    def err_processor(self) -> Processor:
        """ The processor that raised the exception """
        return self._err_processor

    @property
    def processor_type(self) -> str:
        return type(self.err_processor).__name__

    @property
    def processors_list(self) -> list[str]:
        return self._processors_list

    @property
    def provided_reqs(self) -> set[str]:
        return self._provided_reqs

    def build_message(self) -> None:
        self.message = (f"---\nPipeline Requirements Error!\n"
                        f"\tProcessor: {self.processor_type}\n"
                        f"\tPipeline processors list: {','.join(self.processors_list)}\n"
                        f"\tProcessor Requirements: {self.err_processor.requires}\n"
                        f"\t\t- fulfilled: {self.err_processor.requires.intersection(self.provided_reqs)}\n"
                        f"\t\t- missing: {self.err_processor.requires - self.provided_reqs}\n"
                        f"\nThe processors list provided for this pipeline is invalid.  Please make sure all "
                        f"prerequisites are met for every processor.\n\n")

    def __str__(self) -> str:
        return self.message


class Processor(ABC):
    """ Base class for all processors """

    PROVIDES_DEFAULT: ClassVar[set[str]]
    REQUIRES_DEFAULT: ClassVar[set[str]]
    _variant: ProcessorVariant

    def __init__(
            self,
            config,
            pipeline,
            device: ProcessorDevice,
        ) -> None:
        # overall config for the processor
        self._config = config
        # pipeline building this processor (presently processors are only meant to exist in one pipeline)
        self._pipeline = pipeline
        self._set_up_variants(config, device)
        # run set up process
        # set up what annotations are required based on config
        if not self._set_up_variant_requires():
            self._set_up_requires()
        # set up what annotations are provided based on config
        self._set_up_provides()
        # given pipeline constructing this processor, check if requirements are met, throw exception if not
        self._check_requirements()

        if hasattr(self, '_variant') and self._variant.OVERRIDE:
            self.process = self._variant.process

    def __str__(self) -> str:
        """
        Simple description of the processor: name(model)
        """
        name = self.__class__.__name__
        model = None
        if self._config is not None:
            model = self._config.get('model_path')
        if model is None:
            return name
        else:
            return "{}({})".format(name, model)


    @abstractmethod
    def process(self, document: Document, /) -> Document:
        """ Process a Document.  This is the main method of a processor. """
        pass

    def bulk_process(self, docs: list[Document]) -> list[Document]:
        """ Process a list of Documents. This should be replaced with a more efficient implementation if possible. """

        if hasattr(self, '_variant'):
            return self._variant.bulk_process(docs)

        return [self.process(doc) for doc in docs]

    def _set_up_provides(self) -> None:
        """ Set up what processor requirements this processor fulfills.  Default is to use a class defined list. """
        self._provides = self.__class__.PROVIDES_DEFAULT

    def _set_up_requires(self) -> None:
        """ Set up requirements for this processor.  Default is to use a class defined list. """
        self._requires = self.__class__.REQUIRES_DEFAULT

    def _set_up_variant_requires(self) -> bool:
        """
        If this has a variant with its own requirements, use those instead

        Returns True iff the _requires is set from the _variant
        """
        if not hasattr(self, '_variant'):
            return False
        variant = self._variant
        if isinstance(variant, _VariantRequirementsInitializer):
            variant._set_up_requires()
            if not isinstance(variant, _VariantRequirements):
                raise AttributeError(
                    "Processor variant _set_up_requires() did not set _requires"
                )
            self._requires = variant._requires
            return True
        if isinstance(variant, _VariantDefaultRequirements):
            self._requires = variant.REQUIRES_DEFAULT
            return True
        return False

    def _set_up_variants(
            self,
            config,
            device: ProcessorDevice,
        ) -> None:
        processor_name = list(self.__class__.PROVIDES_DEFAULT)[0]
        if any(config.get(f'with_{variant}', False) for variant in PROCESSOR_VARIANTS[processor_name]):
            self._trainer = None
            variant_name = [variant for variant in PROCESSOR_VARIANTS[processor_name] if config.get(f'with_{variant}', False)][0]
            self._variant = PROCESSOR_VARIANTS[processor_name][variant_name](config)

    @property
    def config(self):
        """ Configurations for the processor """
        return self._config

    @property
    def pipeline(self):
        """ The pipeline that this processor belongs to """
        return self._pipeline

    @property
    def provides(self) -> set[str]:
        return self._provides

    @property
    def requires(self) -> set[str]:
        return self._requires

    def _check_requirements(self) -> None:
        """ Given a list of fulfilled requirements, check if all of this processor's requirements are met or not. """
        if not self.config.get("check_requirements", True):
            return
        provided_reqs = set.union(*[processor.provides for processor in self.pipeline.loaded_processors]+[set([])])
        if self.requires - provided_reqs:
            load_names = [item[0] for item in self.pipeline.load_list]
            raise ProcessorRequirementsException(load_names, self, provided_reqs)

    def mark_inactive(self) -> None:
        """Drop optional runtime state when retaining a failed processor."""


class ProcessorVariant(ABC):
    """ Base class for all processor variants """

    OVERRIDE: ClassVar[bool] = False # Set to true to override all the processing from the processor

    def __init__(self, config: Optional[_VariantConfigT] = None) -> None:
        # Variants historically accepted their configuration in subclass
        # constructors.  Keep the base initializer a no-op so third-party
        # variants can call super().__init__() with or without that value.
        pass

    @abstractmethod
    def process(self, doc) -> Document:
        """
        Process a document that is potentially preprocessed by the processor.
        This is the main method of a processor variant.

        If `OVERRIDE` is set to True, all preprocessing by the processor would be bypassed, and the processor variant
        would serve as a drop-in replacement of the entire processor, and has to be able to interpret all the configs
        that are typically handled by the processor it replaces.
        """
        pass

    def bulk_process(self, docs: list[Document]) -> list[Document]:
        """ Process a list of Documents. This should be replaced with a more efficient implementation if possible. """

        return [self.process(doc) for doc in docs]

class UDProcessor(Processor):
    """ Base class for the neural UD Processors (tokenize,mwt,pos,lemma,depparse,sentiment,constituency) """

    _pretrain: Optional[Pretrain]
    _vocab: Optional[ProcessorVocab]

    def __init__(
            self,
            config,
            pipeline,
            device: ProcessorDevice,
        ) -> None:
        super().__init__(config, pipeline, device)

        # UD model resources, set up is processor specific
        self._pretrain = None
        self._trainer = None
        self._vocab = None
        if not hasattr(self, '_variant'):
            self._set_up_model(config, pipeline, device)

        # build the final config for the processor
        self._set_up_final_config(config)

    @abstractmethod
    def _set_up_model(
            self,
            config,
            pipeline,
            device: Optional[ProcessorDevice],
        ) -> None:
        pass

    def _set_up_final_config(self, config) -> None:
        """ Finalize the configurations for this processor, based off of values from a UD model. """
        # set configurations from loaded model
        # Concrete UD processors install different trainer implementations in
        # _set_up_model.  Read the state after that hook has run.
        trainer = getattr(self, "_trainer", None)
        if trainer is not None:
            loaded_args, self._vocab = trainer.args, trainer.vocab
            # filter out unneeded args from model
            loaded_args = {k: v for k, v in loaded_args.items() if not UDProcessor.filter_out_option(k)}
        else:
            loaded_args = {}
        loaded_args.update(config)
        self._config = loaded_args

    def mark_inactive(self) -> None:
        """ Drop memory intensive resources if keeping this processor around for reasons other than running it. """
        self._trainer = None
        self._vocab = None

    @property
    def pretrain(self) -> Optional[Pretrain]:
        return self._pretrain

    @property
    def trainer(self):
        return self._trainer

    @property
    def vocab(self) -> Optional[ProcessorVocab]:
        return self._vocab

    @staticmethod
    def filter_out_option(option: str) -> bool:
        """ Filter out non-processor configurations """
        options_to_filter = ['device', 'cpu', 'cuda', 'dev_conll_gold', 'epochs', 'lang', 'mode', 'save_name', 'shorthand']
        if option.endswith('_file') or option.endswith('_dir'):
            return True
        elif option in options_to_filter:
            return True
        else:
            return False

    def bulk_process(self, docs: list[Document]) -> list[Document]:
        """
        Most processors operate on the sentence level, where each sentence is processed independently and processors can benefit
        a lot from the ability to combine sentences from multiple documents for faster batched processing. This is a transparent
        implementation that allows these processors to batch process a list of Documents as if they were from a single Document.
        """

        if hasattr(self, '_variant'):
            return self._variant.bulk_process(docs)

        combined_sents = [sent for doc in docs for sent in doc.sentences]
        combined_doc = Document([])
        combined_doc.sentences = combined_sents
        combined_doc.num_tokens = sum(doc.num_tokens for doc in docs)
        combined_doc.num_words = sum(doc.num_words for doc in docs)

        self.process(combined_doc) # annotations are attached to sentence objects

        return docs

class ProcessorRegisterException(Exception):
    """ Exception indicating processor or processor registration failure """

    def __init__(self, processor_class, expected_parent):
        self._processor_class = processor_class
        self._expected_parent = expected_parent
        self.build_message()

    def build_message(self) -> None:
        self.message = f"Failed to register '{self._processor_class}'. It must be a subclass of '{self._expected_parent}'."

    def __str__(self) -> str:
        return self.message

def register_processor(
        name: str,
    ) -> Callable[[type[_ProcessorT]], type[_ProcessorT]]:
    def wrapper(Cls: type[_ProcessorT]) -> type[_ProcessorT]:
        if not issubclass(Cls, Processor):
            raise ProcessorRegisterException(Cls, Processor)

        NAME_TO_PROCESSOR_CLASS[name] = Cls
        PIPELINE_NAMES.append(name)
        return Cls
    return wrapper

def register_processor_variant(
        name: str,
        variant: str,
    ) -> Callable[[type[_ProcessorVariantT]], type[_ProcessorVariantT]]:
    def wrapper(
            Cls: type[_ProcessorVariantT],
        ) -> type[_ProcessorVariantT]:
        if not issubclass(Cls, ProcessorVariant):
            raise ProcessorRegisterException(Cls, ProcessorVariant)

        PROCESSOR_VARIANTS[name][variant] = Cls
        return Cls
    return wrapper
