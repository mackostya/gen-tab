# Tabular data generation

This project solves as an example ML / DL project for tabular data generation. It takes at the basis the approach of [CTGAN and TVAE](https://github.com/sdv-dev/CTGAN) and improves the framework in follwoing context:

- Uses the approach of firstly fitting the multivariate gaussians before training the VAE and GAN models
- Framework rewritten in PyTorch Lightning -> more flexible and easily manageble
- Implemented the Tensorboard pluggin to visualize the training processes

# Data

An example dataset will be used for the data generation pipeline. The dataset is taken from one of the most popular kaggle challanges: [Titanic](https://www.kaggle.com/competitions/titanic/data). Download and put it into the folder `data`.

# Environment

Create environment as follows:

```
conda env create -f environment.yml
```