from abc import ABC, abstractmethod
from lightning.pytorch import loggers as pl_loggers
import matplotlib.pyplot as plt
from .tools import load_cfg
import src.visualizations as vis
import pandas as pd


class ModelInterface(ABC):
    def __init__(self):
        super().__init__()
        self._tb_settings = load_cfg("conf/tb_settings.yaml")
        self.tb_logger = None

    @abstractmethod
    def load(self, file_path):
        """
        Pseudo-Interface for loading trained models. The explicit loading has to be defined in the corresponding class.
        :param file_path: (string) Path to the saved model
        :return:
        """
        pass

    @abstractmethod
    def generate(self, n_samples_to_generate):
        """
        Pseudo-Interface for generating samples. The explicit generation has to be defined in the corresponding class.
        :n_samples_to_generate: (int) Number of samples to generate
        :return:
        """
        pass

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

    def log_tb_distributions(self, real, fake) -> None:
        """
        Interface for logging distributions to tensorboard.
        :return:
        """
        if isinstance(fake, pd.DataFrame):
            fake = fake.to_numpy()
        for feature_id in self._tb_settings["features_to_visualize"]:
            real_to_plot = self._transformer.inverse_transform(real)
            if isinstance(real_to_plot, pd.DataFrame):
                real_to_plot = real_to_plot.to_numpy()
            fig, _ = vis.hist_real_vs_fake_plot(
                real_data=real_to_plot[:, feature_id],
                fake_data=fake[:, feature_id],
                bins=50,
                name=f"Feature {feature_id}",
            )
            self.tb_logger.add_figure(f"Image/{feature_id}", fig, global_step=self.current_epoch)
            plt.close(fig)
