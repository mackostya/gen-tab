import torch
import lightning as L
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from src.dataset.data_transformer import DataTransformer
from .model.unet import UNet

from .scheduler.ddim import DDIMScheduler
from ..loop_interface import ModelInterface


class DiffusionLoop(ModelInterface, L.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        n_timesteps: int,
        batch_size: int,
        data_transformer: DataTransformer,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.input_dim = input_dim
        self._transformer = data_transformer
        self.batch_size = batch_size
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=n_timesteps,
            beta_schedule="cosine",
            beta_start=0.0001,
            beta_end=0.02,
        )
        self.unet = UNet(input_size=input_dim, hidden_dims=hidden_dims, use_linear_attn=False)

    def _step(self, x):
        x = x.unsqueeze(1).long()
        batch_size = x.shape[0]
        noise = torch.randn(x.shape).to(self.device)

        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps, (batch_size,), device=self.device
        ).long()
        noisy_x = self.noise_scheduler.add_noise(x, noise, timesteps).to(self.device)
        noise_pred = self.unet(noisy_x, timesteps)["sample"]
        loss = F.l1_loss(noise_pred, noise)
        return loss

    def training_step(self, batch, batch_idx):
        x = batch
        x = torch.Tensor(x).to(self.device)
        loss = self._step(x)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x = batch
        x = torch.Tensor(x).to(self.device)
        loss = self._step(x)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        if self.current_epoch % self._tb_settings["tensorboard_visualize_epochs"] == 0 and batch_idx == 0:
            fake = self.generate(x.shape[0])
            self.log_tb_distributions(x.detach().cpu().numpy(), fake)

        return {"loss": loss}

    @classmethod
    def load(self, path):
        pass

    def forward(self, inputs, timesteps):
        return self.unet(inputs, timesteps)

    def generate(self, n_samples_to_generate, num_inference_steps=50):
        generator = torch.manual_seed(0)
        n_batches = n_samples_to_generate // self.batch_size
        n_rest = n_samples_to_generate % self.batch_size
        batch_sizes = np.ones(n_batches + 1, dtype=int) * self.batch_size
        batch_sizes[-1] = n_rest
        output = np.zeros((0, self.input_dim))
        for b in tqdm(batch_sizes):
            out = self.noise_scheduler.generate(
                model=self.unet,
                generator=generator,
                batch_size=b,
                num_inference_steps=num_inference_steps,
                device=self.device,
            )
            sample = out["sample"].squeeze().cpu().numpy()
            output = np.concatenate((output, sample), axis=0)

        return output

    def configure_optimizers(self):
        self.init_tb_logger()
        return torch.optim.Adam(
            self.unet.parameters(),
            lr=1e-4,
            betas=(0.95, 0.999),
            weight_decay=1e-5,
        )
