"""
base classes for processors
"""

from abc import ABC, abstractmethod


# base class for all processors
class Processor(ABC):

    @abstractmethod
    def process(self, doc):
        pass
