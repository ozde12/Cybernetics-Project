from abc import ABC, abstractmethod

class AudioAlgorithm(ABC):
    name = "base"

    @abstractmethod
    def process(self, features: dict, memory: dict) -> dict:
        """
        Returns: {"audio": np.ndarray [float32, shape=(N,)], "sr": int, "meta": dict}
        """
        raise NotImplementedError
