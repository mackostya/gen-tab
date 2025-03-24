import logging
import lightning as L
import os
import torch
import click

from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint

from src.factory import GenerationModelFactory
from src.dataset.house_pricing_dataset import HousePricingDataset
from src.dataset.data_transformer import DataTransformer
from src.tools import load_cfg, init_logging, load_data_transformer, save_data_transformer


def load_data(cfg):

    data_transformer = DataTransformer()

    if os.path.exists(cfg["data_transformer_path"]):
        logging.info("Loading data transformer from file.")
        data_transformer = load_data_transformer(cfg["data_transformer_path"])

        train_dataset = HousePricingDataset(cfg["train_data_path"], data_transformer=data_transformer)

    else:
        logging.info("No data transformer file found. Fitting data transformer.")
        train_dataset = HousePricingDataset(cfg["train_data_path"], data_transformer=data_transformer)
        train_dataset.fit_transformer()

        save_data_transformer(data_transformer, cfg["data_transformer_path"])

    train_dataset.transform()

    val_dataset = HousePricingDataset(cfg["val_data_path"], data_transformer=data_transformer)
    val_dataset.transform()
    init_data_dim = train_dataset.input_dim
    sample_input = next(iter(train_dataset))
    input_size_after_transformation = sample_input.shape[0]

    logging.info(f"Validation dataset: {len(val_dataset)} samples. Training dataset: {len(train_dataset)} samples.")
    logging.info(f"Data of shape {init_data_dim} loaded. Transformed to {input_size_after_transformation}.")
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        # num_workers=21,  # define depending on the system
        shuffle=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        # num_workers=21,  # define depending on the system
        # persistent_workers=True,
        shuffle=False,
    )

    return train_dataloader, val_dataloader, data_transformer, init_data_dim


@click.command()
@click.option(
    "--model-type",
    default="vae",
    help="Type of model to train. Either 'vae' (default) or 'gan'.",
)
@click.option(
    "--resume-from",
    default=None,
    help="If you want to resume training from a checkpoint, provide the path to the checkpoint.",
)
def main(model_type: str, resume_from: str):
    torch.set_float32_matmul_precision("high")

    cfg = load_cfg()

    log_dir = init_logging(model_type, resume_from, additional_training_comment="some_comment")

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir, name="tb_logger")

    logging.info(f"{'Initializing data':-^{50}}")

    train_dataloader, val_dataloader, data_transformer, init_data_dim = load_data(cfg)

    if model_type == "vae":
        ckeckpoint = ModelCheckpoint(
            dirpath=log_dir,
            filename="{epoch}-{val_loss:.2f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )
    trainer = L.Trainer(
        max_epochs=cfg["epochs"],
        accelerator="gpu",
        devices=[0],
        logger=tb_logger,
        callbacks=[ckeckpoint] if model_type == "vae" else None,
        log_every_n_steps=10,
    )
    model_factory = GenerationModelFactory(
        model_type=model_type,
        cfg=cfg,
        input_dim=init_data_dim,
        data_transformer=data_transformer,
        resume_from=resume_from,
    )
    trainer_loop = model_factory.get_model()

    trainer.fit(model=trainer_loop, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    logging.info(f"Saving done.")


if __name__ == "__main__":
    main()
