import abc
import torch
from typing import List, Dict

class Featurizer(abc.ABC):
    """
    Abstract base class for a featurizer. Defines the standard interface
    that all concrete featurizers must implement.
    """
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """A unique, directory-safe name for the featurizer."""
        pass

    @abc.abstractmethod
    def featurize(self, items: List[str]) -> Dict[str, torch.Tensor]:
        """
        Takes a list of items (strings) and returns a dictionary mapping
        each item to its computed feature tensor.
        """
        pass