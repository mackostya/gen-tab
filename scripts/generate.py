import logging
import torch
import click
import os

# fix QT Issue for debugging
import matplotlib as mpl

mpl.use("Qt5Agg")

from src.factory import GenerationModelFactory
from src.tools import load_cfg


@click.command()
@click.option(
    "--model-type",
    default="vae",
    help="Type of model to generate data from, 'vae' (default) or 'gan'.",
)
def main(model_type):
    torch.set_float32_matmul_precision("high")
    cfg = load_cfg()
    n_samples_to_generate = cfg[model_type]["generation"]["n_samples_to_generate"]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(threadName)-10s) %(message)s",
        force=True,
    )

    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    logging.info("Loading model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    factory = GenerationModelFactory(
        model_type=model_type,
        cfg=cfg,
        resume_from=cfg[model_type]["generation"]["model_path"],
    )
    model = factory.get_model()
    model.to(device)
    model.eval()

    logging.info(f"Input dimension {model._input_dim}")
    logging.info("Generating data...")

    with torch.no_grad():
        logging.info(f"using pretrained model")
        generated_data = model.generate(
            n_samples_to_generate=n_samples_to_generate,
        )
        logging.info(generated_data.shape)

    if not os.path.exists(cfg["data_dir"]):
        os.makedirs(cfg["data_dir"])

    generated_data.to_csv(cfg["data_dir"] + cfg[model_type]["generation"]["new_generated_data_filename"])
    logging.info("Data generated.")


if __name__ == "__main__":
    main()
