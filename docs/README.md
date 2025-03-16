# CTGAN and TVAE License

The core of this repository is built on the [CTGAN](https://github.com/sdv-dev/CTGAN/blob/main/ctgan/synthesizers/ctgan.py) and [TVAE](https://github.com/sdv-dev/CTGAN/blob/main/ctgan/synthesizers/tvae.py) implementations. The main model, built with [PyTorch](https://pytorch.org/), has not been altered in order to maintain reproducibility. The [DataTransformer](https://github.com/sdv-dev/CTGAN/blob/main/ctgan/data_transformer.py) implementation was slightly adapted to fix logging issues and provide more detailed information to the trained model.

For more information on how to reproduce CTGAN and TVAE, see their [license](https://github.com/sdv-dev/CTGAN/blob/main/LICENSE).
