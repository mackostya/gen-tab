from abc import ABC, abstractmethod


class NoiseSchdulerInterface(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def add_noise(self, x, noise, timesteps):
        pass
