"""
Module which specifies high level Annotator design
"""

from abc import ABC, abstractmethod

class Annotator(ABC):
    """Abstract class representing a generic NLP annotator"""

    @abstractmethod
    def annotate(self, doc):
        """Add this annotator's annotations to the document"""
        pass

    @abstractmethod
    def requires(self):
        """Return a set of annotation requirements for this annotator"""
        pass

    @abstractmethod
    def requirements_satisfied(self):
        """Return what annotations this annotator will provide"""
        pass
