"""
VAE training loop.
Adapted from https://github.com/sdv-dev/CTGAN/blob/main/ctgan/synthesizers/tvae.py
"""

from typing import Optional, Union
import torch
import logging
import warnings

import lightning as L
import numpy as np
from src.dataset.data_transformer import DataTransformer
from lightning.pytorch import loggers as pl_loggers
from .vae import Encoder, Decoder, loss_function
from ..tools import load_cfg


class VAELoop(L.LightningModule):
    """
    VAE training loop class that handles the training and validation steps, as well as data generation.

    Args:
        input_dim (int): The dimension of the input data, before the data transformation.
        features_type (str): The type of features (e.g., "analog").
        embedding_dim (int): The dimension of the embedding space.
        compress_dims (list): The dimensions of the layers for the encoder.
        decompress_dims (list): The dimensions of the layers for the decoder.
        batch_size (int): The size of the training batches.
        data_transformer (DataTransformer): The data transformer used for preprocessing.

    Attributes:
        _embedding_dim (int): The dimension of the embedding space.
        _compress_dims (list): The dimensions of the layers for the encoder.
        _decompress_dims (list): The dimensions of the layers for the decoder.
        _batch_size (int): The size of the training batches.
        _transformer (DataTransformer): The data transformer used for preprocessing.
        _input_dim (int): The dimension of the input data.
        _l2scale (float): The L2 regularization scale.
        _loss_factor (float): The loss factor for the KL divergence.
        _encoder (Encoder): The encoder model.
        _decoder (Decoder): The decoder model.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        compress_dims: list,
        decompress_dims: list,
        batch_size: int,
        data_transformer: DataTransformer,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self._embedding_dim = embedding_dim
        self._compress_dims = compress_dims
        self._decompress_dims = decompress_dims
        self._batch_size = batch_size
        self._transformer = data_transformer
        self._input_dim = input_dim

        self._l2scale = 1e-4
        self._loss_factor = 2.0

        self.tb_logger = None
        data_dim = self._transformer.output_dimensions

        self._encoder = Encoder(
            data_dim,
            self._compress_dims,
            self._embedding_dim,
        ).to(self._device)

        self._decoder = Decoder(
            self._embedding_dim,
            self._decompress_dims,
            data_dim,
        ).to(self._device)
        self._tb_settings = load_cfg("conf/tb_settings.yaml")
        warnings.simplefilter(action="ignore", category=FutureWarning)

    def _step(self, x):
        """
        Perform a single training step.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            tuple: The reconstruction loss, KL divergence loss, and embedding.
        """
        out = self(x)
        loss_rec, loss_kld = loss_function(
            out["rec"],
            x,
            out["sigmas"],
            out["mu"],
            out["logvar"],
            self._transformer.output_info_list,
            self._loss_factor,
        )
        return loss_rec, loss_kld, out["emb"]

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.

        Args:
            batch (tuple): The input batch.
            batch_idx (int): The batch index.

        Returns:
            dict: The training losses.
        """
        vae_opt = self.optimizers()

        x = batch

        real = x.to(torch.float32).to(self._device)

        loss_rec, loss_kld, _ = self._step(real)

        loss = loss_rec + loss_kld
        vae_opt.zero_grad(set_to_none=True)
        self.manual_backward(loss)
        vae_opt.step()
        self._decoder.sigma.data.clamp_(0.01, 1.0)

        self.log("train_rec_loss", loss_rec, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_kld_loss", loss_kld, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        if self.current_epoch % self._tb_settings["tensorboard_visualize_epochs"] == 0 and batch_idx == 0:
            self.log_tb_histograms()
        return {"loss": loss, "loss_rec": loss_rec, "loss_kld": loss_kld}

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.

        Args:
            batch (tuple): The input batch.
            batch_idx (int): The batch index.

        Returns:
            dict: The validation losses.
        """
        x = batch

        real = x.to(torch.float32).to(self._device)

        loss_rec, loss_kld, emb = self._step(real)

        loss = loss_rec + loss_kld

        self.log("val_rec_loss", loss_rec, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_kld_loss", loss_kld, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        # if self.current_epoch % self._tb_settings["tensorboard_visualize_epochs"] == 0 and batch_idx == 0:
        #     fake = self.generate(x.shape[0])
        # self.log_tb_distributions(real.detach().cpu().numpy(), fake)
        # self.log_tb_correlations(fake)
        return {"loss": loss, "loss_rec": loss_rec, "loss_kld": loss_kld}

    @classmethod
    def load(cls, file_path):
        """
        Load a VAE model from a checkpoint.

        Args:
            file_path (str): The path to the checkpoint file.

        Returns:
            VAELoop: The loaded VAE model.
        """
        checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)
        logging.info(f"Loading model with hyperparameters: {checkpoint['hyper_parameters']}")
        return VAELoop.load_from_checkpoint(file_path, checkpoint["hyper_parameters"])

    def generate(self, n_samples_to_generate):
        """
        Sample data similar to the training data.

        Args:
            n_samples_to_generate (int): Number of rows to sample.

        Returns:
            numpy.ndarray: The generated data.
        """

        steps = n_samples_to_generate // self._batch_size + 1
        data = []
        for _ in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self._device)

            fake, sigmas = self._decoder(noise)
            fake = torch.tanh(fake)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n_samples_to_generate]
        return self._transformer.inverse_transform(data, sigmas.detach().cpu().numpy())

    def forward(self, inputs):
        """
        Forward pass through the model.

        Args:
            inputs (torch.Tensor): The input data.

        Returns:
            dict: The model output including reconstruction, sigmas, mu, logvar, and embedding.
        """
        x = inputs.to(torch.float32)
        mu, std, logvar = self._encoder(x)
        eps = torch.randn_like(std)
        emb = eps * std + mu
        x = emb.to(torch.float32)
        x, sigmas = self._decoder(x)

        return {"rec": x, "sigmas": sigmas, "mu": mu, "logvar": logvar, "emb": emb}

    def init_tb_logger(self):
        if self.tb_logger is None:
            # Get tensorboard logger
            tb_logger = None
            for logger in self.trainer.loggers:
                if isinstance(logger, pl_loggers.TensorBoardLogger):
                    tb_logger = logger.experiment
                    break

            if tb_logger is None:
                raise ValueError("TensorBoard Logger not found")
            self.tb_logger = tb_logger
            return self.tb_logger
        else:
            return self.tb_logger

    def log_tb_histograms(self) -> None:
        """
        Interface for logging histograms to tensorboard.
        :return:
        """
        for block_name in self._tb_settings["parameters_to_visualize"]:
            subblock_names = block_name.split(".")
            block = self
            for b_name in subblock_names:
                block = getattr(block, b_name)
            for name, weight in block.named_parameters():
                self.tb_logger.add_histogram(name, weight, self.current_epoch)
                if weight.grad is not None:
                    self.tb_logger.add_histogram(name + "_grad", weight.grad, self.current_epoch)

    def configure_optimizers(self):
        """
        Configure the optimizers for the model.

        Returns:
            torch.optim.Optimizer: The optimizer for the model.
        """
        self.init_tb_logger()
        vae_opt = torch.optim.Adam(
            list(self._encoder.parameters()) + list(self._decoder.parameters()),
            weight_decay=self._l2scale,
        )
        return vae_opt
