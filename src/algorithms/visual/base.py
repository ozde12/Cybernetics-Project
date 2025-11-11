from abc import ABC, abstractmethod

class VisualAlgorithm(ABC):
    name = "base"

    @abstractmethod
    def draw(self, features: dict, memory: dict):
        """
        Return either a numpy RGB image or draw directly to a matplotlib Axes you manage.
        """
        raise NotImplementedError
