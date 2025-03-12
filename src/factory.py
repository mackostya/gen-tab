import torch
import random
import numpy as np

from dataclasses import dataclass
from typing import Optional
from src.vae.loop import VAELoop
from src.gan.loop import GANLoop
from src.dataset.data_transformer import DataTransformer


@dataclass
class GenerationModelFactory:
    """
    Factory class for creating and configuring generative models (GANs or VAEs).

    Args:
        model_type (str): The type of the model to create ('gan' or 'vae').
        cfg (dict): Configuration dictionary for the model.
        data_transformer (Optional[DataTransformer]): The data transformer used for preprocessing.
        input_dim (int): The dimension of the input data (default is 38).
        resume_from (Optional[str]): Path to a checkpoint to resume training from (default is None).
        seed (int): Random seed for reproducibility (default is 1234).
    """

    model_type: str
    cfg: dict
    data_transformer: Optional[DataTransformer] = DataTransformer()
    input_dim: int = 38
    resume_from: Optional[str] = None
    seed: int = 1234

    def __post_init__(self):
        """
        Initialize the factory by setting random seeds for reproducibility.
        """
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

    def get_model(self):
        """
        Create and return the appropriate generative model based on the configuration.

        Returns:
            GANLoop or VAELoop: The initialized generative model.
        """
        self._check_setup()
        if self.model_type == "vae":
            return self._get_vae()
        elif self.model_type == "gan":
            return self._get_gan()
        else:
            raise ValueError(f"Unknown model type: {self.cfg['model_type']}")

    def _check_setup(self):
        """
        Check if the data transformer has been fitted. Raise an error if not.
        """
        if self.data_transformer.output_dimensions == 0 and self.resume_from == None:
            raise ValueError("Data transformer not fitted. Please fit the data transformer before training the model.")

    def _get_vae(self):
        """
        Create and return a VAE model. Load from checkpoint if specified.

        Returns:
            VAELoop: The initialized VAE model.
        """
        if self.resume_from is None:
            return VAELoop(
                input_dim=self.input_dim,
                embedding_dim=self.cfg[self.model_type]["embedding_dim"],
                compress_dims=self.cfg[self.model_type]["compress_dims"],
                decompress_dims=self.cfg[self.model_type]["decompress_dims"],
                batch_size=self.cfg["batch_size"],
                data_transformer=self.data_transformer,
            )
        else:
            return self._get_vae_from_checkpoint()

    def _get_gan(self):
        """
        Create and return a GAN model. Load from checkpoint if specified.

        Returns:
            GANLoop: The initialized GAN model.
        """
        if self.resume_from is None:
            return GANLoop(
                input_dim=self.input_dim,
                discriminator_dim=self.cfg[self.model_type]["discriminator_dims"],
                generator_dim=self.cfg[self.model_type]["generator_dims"],
                embedding_dim=self.cfg[self.model_type]["embedding_dim"],
                batch_size=self.cfg["batch_size"],
                data_transformer=self.data_transformer,
            )
        else:
            return self._get_gan_from_checkpoint()

    def _get_vae_from_checkpoint(self):
        """
        Load a VAE model from a checkpoint.

        Returns:
            VAELoop: The loaded VAE model.
        """
        return VAELoop.load(self.cfg["model_path"] if self.resume_from is None else self.resume_from)

    def _get_gan_from_checkpoint(self):
        """
        Load a GAN model from a checkpoint.

        Returns:
            GANLoop: The loaded GAN model.
        """
        return GANLoop.load(self.cfg["model_path"] if self.resume_from is None else self.resume_from)
