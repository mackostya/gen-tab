import torch, torch.nn as nn, torch.utils.data as data, torchvision as tv
import lightning as L
import numpy as np
from lightning.pytorch import loggers as pl_loggers
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from .model.unet import UNet

from .scheduler.ddim import DDIMScheduler


class DiffusionLoop(L.LightningModule):
    def __init__(
        self,
        input_size: int,
        hidden_dims: list,
        n_timesteps: int,
    ):
        super().__init__()
        self.save_hyperparameters()

        # TODO: Load data transformer if needed
        self.data_transformer = ...

        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=n_timesteps,
            beta_schedule="cosine",
            beta_start=0.0001,
            beta_end=0.02,
        )
        self.unet = UNet(input_size=input_size, hidden_dims=hidden_dims, use_linear_attn=False)

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
        x, _ = batch
        x = torch.Tensor(x).to(self.device)
        loss = self._step(x)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x = torch.Tensor(x).to(self.device)
        loss = self._step(x)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        if self.current_epoch % 2 == 0 and batch_idx == 0:
            self.log_tb_images(x, "LS_TS1_Energy")
            self.log_tb_images(x, "LS_TS2")

        return {"loss": loss}

    def log_tb_images(self, x, feature_name) -> None:
        # Get tensorboard logger
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, pl_loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
            raise ValueError("TensorBoard Logger not found")
        generator = torch.manual_seed(0)
        out = self.noise_scheduler.generate(
            model=self.unet,
            generator=generator,
            batch_size=x.shape[0],
            num_inference_steps=50,
            eta=0.0,
            save_intermediate_timesteps=True,
            device=self.device,
        )

        timesteps = sorted(out["intermediate_samples"].keys())
        model_out = torch.stack([out["intermediate_samples"][t] for t in timesteps], dim=0)
        noise = {t: torch.randn_like(x) for t in timesteps}
        noisy_out = torch.stack(
            [
                self.noise_scheduler.add_noise(
                    x,
                    noise[t],
                    torch.ones(x.shape[0], dtype=torch.long) * t,
                )
                for t in timesteps
            ],
            dim=0,
        )

        # TODO: implement visualization of model behaviour
        fig = ...
        tb_logger.add_figure(f"Image/{feature_name}", fig, global_step=self.current_epoch)
        plt.close(fig)

        noise_out = torch.stack(
            [out["model_output"][t].squeeze() for t in timesteps],
            dim=0,
        )
        # TODO: implement visualization of noise development
        fig = ...
        tb_logger.add_figure(f"Image/Noise_of_{feature_name}", fig, global_step=self.current_epoch)
        plt.close(fig)

    def forward(self, inputs, timesteps):
        return self.unet(inputs, timesteps)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.unet.parameters(),
            lr=1e-4,
            betas=(0.95, 0.999),
            weight_decay=1e-5,
        )

    def generate(self, n_samples_to_generate, num_inference_steps, batch_size=512):
        generator = torch.manual_seed(0)
        n_batches = n_samples_to_generate // batch_size
        n_rest = n_samples_to_generate % batch_size
        batch_sizes = np.ones(n_batches + 1, dtype=int) * batch_size
        batch_sizes[-1] = n_rest
        output = np.zeros((0, self.hparams.input_size))
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
