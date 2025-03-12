"""
GAN training loop.
Adapted from https://github.com/sdv-dev/CTGAN/blob/main/ctgan/synthesizers/ctgan.py
"""

import torch
import logging

import lightning as L
import numpy as np

from torch.nn import functional
from lightning.pytorch import loggers as pl_loggers

from .model.generator import Generator
from .model.discriminator import Discriminator

from src.dataset.data_transformer import DataTransformer
from ..tools import load_cfg


class GANLoop(L.LightningModule):
    """
    GAN training loop class that handles the training and validation steps, as well as data generation.

    Args:
        input_dim (int): The dimension of the input data.
        discriminator_dim (list): The dimensions of the layers for the discriminator.
        generator_dim (list): The dimensions of the layers for the generator.
        embedding_dim (int): The dimension of the embedding space.
        batch_size (int): The size of the training batches.
        data_transformer (DataTransformer): The data transformer used for preprocessing.

    Attributes:
        pac (int): The number of samples to use for pacGAN.
        _generator_lr (float): The learning rate for the generator.
        _generator_decay (float): The weight decay for the generator.
        _discriminator_lr (float): The learning rate for the discriminator.
        _discriminator_decay (float): The weight decay for the discriminator.
        _batch_size (int): The size of the training batches.
        _embedding_dim (int): The dimension of the embedding space.
        _input_dim (int): The dimension of the input data.
        _transformer (DataTransformer): The data transformer used for preprocessing.
        _discriminator (Discriminator): The discriminator model.
        _generator (Generator): The generator model.
    """

    def __init__(
        self,
        input_dim: int,
        discriminator_dim: list,
        generator_dim: list,
        embedding_dim: int,
        batch_size: int,
        data_transformer: DataTransformer,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.pac = 1
        self._generator_lr = 1e-5
        self._generator_decay = 1e-6
        self._discriminator_lr = 5e-5
        self._discriminator_decay = 1e-6

        self.tb_logger = None
        self._batch_size = batch_size
        self._embedding_dim = embedding_dim
        self._input_dim = input_dim

        self._transformer = data_transformer
        data_dim = self._transformer.output_dimensions

        self._discriminator = Discriminator(input_dim=data_dim, discriminator_dim=discriminator_dim, pac=self.pac).to(
            self.device
        )
        self._generator = Generator(embedding_dim=embedding_dim, generator_dim=generator_dim, data_dim=data_dim).to(
            self.device
        )
        self._tb_settings = load_cfg("conf/tb_settings.yaml")

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """
        Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

        Args:
            logits (torch.Tensor): Unnormalized log probabilities.
            tau (float): Non-negative scalar temperature.
            hard (bool): If True, the returned samples will be discretized as one-hot vectors.
            eps (float): Small value to avoid division by zero.
            dim (int): A dimension along which softmax will be computed.

        Returns:
            torch.Tensor: Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        for _ in range(10):
            transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
            if not torch.isnan(transformed).any():
                return transformed

        raise ValueError("gumbel_softmax returning NaN.")

    def _apply_activate(self, data):
        """
        Apply the proper activation function to the output of the generator.

        Args:
            data (torch.Tensor): The generator output.

        Returns:
            torch.Tensor: The activated generator output.
        """
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == "tanh":
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == "softmax":
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f"Unexpected activation function {span_info.activation_fn}.")

        return torch.cat(data_t, dim=1)

    def _discriminator_step(self, x, train_step=True):
        """
        Perform a single step for the discriminator.

        Args:
            x (torch.Tensor): The input data.
            train_step (bool): Whether to update the discriminator weights.

        Returns:
            torch.Tensor: The discriminator loss.
        """
        _, d_opt = self.optimizers()

        out = self(x)
        loss_d = -(torch.mean(out["y_real"]) - torch.mean(out["y_fake"]))

        if train_step:
            pen = self._discriminator.calc_gradient_penalty(
                x.to(torch.float32),
                out["fake"].to(torch.float32),
                self._device,
                self.pac,
            )
            d_opt.zero_grad(set_to_none=True)
            self.manual_backward(loss_d, retain_graph=True)
            self.manual_backward(pen, retain_graph=True)
            d_opt.step()
        return loss_d

    def _generator_step(self, x, train_step=True):
        """
        Perform a single step for the generator.

        Args:
            x (torch.Tensor): The input data.
            train_step (bool): Whether to update the generator weights.

        Returns:
            tuple: The generator loss and generated samples.
        """
        g_opt, _ = self.optimizers()

        out = self(x)
        loss_g = -torch.mean(out["y_fake"])
        if train_step:
            g_opt.zero_grad(set_to_none=False)
            self.manual_backward(loss_g)
            g_opt.step()
        return loss_g, out["fake"]

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.

        Args:
            batch (tuple): The input batch.
            batch_idx (int): The batch index.

        Returns:
            dict: The training losses.
        """
        x = batch
        x = x.to(self._device).to(torch.float32)

        loss_d = self._discriminator_step(x)
        loss_g, _ = self._generator_step(x)

        self.log("train_loss_d", loss_d, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_loss_g", loss_g, on_step=False, on_epoch=True, sync_dist=True)
        if self.current_epoch % self._tb_settings["tensorboard_visualize_epochs"] == 0 and batch_idx == 0:
            self.log_tb_histograms()
        return {"loss_d": loss_d, "loss_g": loss_g}

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
        x = x.to(self._device).to(torch.float32)

        loss_d = self._discriminator_step(x, train_step=False)
        loss_g, fake_t = self._generator_step(x, train_step=False)

        self.log("val_loss_d", loss_d, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_loss_g", loss_g, on_step=False, on_epoch=True, sync_dist=True)
        if self.current_epoch % self._tb_settings["tensorboard_visualize_epochs"] == 0 and batch_idx == 0:
            fake_t = fake_t.detach().cpu().numpy()
            fake = self.generate(x.shape[0])

        return {"loss_d": loss_d, "loss_g": loss_g}

    def generate(self, n_samples_to_generate: int):
        """
        Generate synthetic data.

        Args:
            n_samples_to_generate (int): The number of samples to generate.

        Returns:
            np.ndarray: The generated samples.
        """
        n_iter = n_samples_to_generate // self._batch_size + 1
        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device).to(self._device)
        std = mean + 1
        output = np.zeros([0, self._input_dim])
        for k in range(n_iter):
            fakez = torch.normal(mean=mean, std=std).to(self._device).to(torch.float32)
            fake = self._generator(fakez)

            fake = self._apply_activate(fake)
            fake = fake.detach().cpu().numpy()
            sample = self._transformer.inverse_transform(fake)
            output = np.concatenate((output, sample), axis=0)
        output = output[:n_samples_to_generate]
        assert (
            output.shape[0] == n_samples_to_generate
        ), f"Wrong number of samples generated. {output.shape} != {(n_samples_to_generate, self._input_dim)}"
        return output

    @classmethod
    def load(cls, file_path):
        """
        Load a model from a checkpoint.

        Args:
            file_path (str): The path to the checkpoint file.

        Returns:
            GANLoop: The loaded model.
        """
        checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)
        logging.info(f"Loading model with hyperparameters: {checkpoint['hyper_parameters']}")
        return GANLoop.load_from_checkpoint(file_path, checkpoint["hyper_parameters"])

    def forward(self, inputs):
        """
        Forward pass through the model.

        Args:
            inputs (torch.Tensor): The input data.

        Returns:
            dict: The model output including fake data, discriminator outputs for fake and real data, and label embeddings.
        """
        mean = torch.zeros(inputs.shape[0], self._embedding_dim, device=self._device).to(self._device)
        std = mean + 1
        fakez = torch.normal(mean=mean, std=std).to(self._device).to(torch.float32)

        fake = self._generator(fakez)
        fake = self._apply_activate(fake)
        real = inputs

        y_fake = self._discriminator(fake.to(torch.float32))
        y_real = self._discriminator(real.to(torch.float32))

        return {"fake": fake, "y_fake": y_fake, "y_real": y_real}

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
            tuple: The optimizers for the generator and discriminator.
        """
        self.init_tb_logger()
        g_opt = torch.optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )
        d_opt = torch.optim.Adam(
            self._discriminator.parameters(),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )
        return g_opt, d_opt
