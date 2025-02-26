import yaml
import datetime
import logging
import os
import pickle


def init_logging(model_type: str, resume_from: str = None, additional_training_comment: str = "") -> str:
    """
    Initializes the logging configuration and creates a directory for log files.

    Args:
        model_type (str): Type of the model (e.g., "vae", "gan").
        resume_from (str): Path from where the model training is resumed, if applicable. Default is None.
        additional_training_comment (str): Additional comments to include in the log directory name. Default is an empty string.

    Returns:
        str: The path to the log directory.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(threadName)-10s) %(message)s",
        force=True,
    )
    iso_date = datetime.datetime.today()

    log_dir = (
        os.path.abspath(os.curdir)
        + f"/logs/{model_type}/{iso_date.isocalendar()[0]}/{iso_date.strftime('%B')}/cw{iso_date.isocalendar()[1]}_day{iso_date.weekday()}_{iso_date.hour}-{iso_date.minute:0>2}_{additional_training_comment}/"
    )

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    if resume_from is not None:
        with open(log_dir + "resumed_from.txt", "w") as file:
            file.write("Resumed from: " + resume_from + "\n")

    return log_dir


def load_cfg(path: str = "conf/generation_config.yaml"):
    """
    Loads the configuration file in YAML format.

    Args:
        path (str): Path to the YAML configuration file. Default is "conf/generation_config.yaml".

    Returns:
        dict: The loaded configuration as a dictionary.
    """
    return yaml.safe_load(open(path))


def save_data_transformer(obj, filename):
    """
    Saves a data transformer object to a file using pickle.

    Args:
        obj: The data transformer object to be saved.
        filename (str): The path to the file where the object will be saved.
    """
    if not os.path.exists(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_data_transformer(filename):
    """
    Loads a data transformer object from a file using pickle.

    Args:
        filename (str): The path to the file from which the object will be loaded.

    Returns:
        The loaded data transformer object.
    """
    with open(filename, "rb") as inp:
        data_transformer = pickle.load(inp)
    return data_transformer
