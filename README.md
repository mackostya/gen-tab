# Tabular data generation

This project serves as an example ML / DL project for tabular data generation. It takes at the basis the approach of [CTGAN and TVAE](https://github.com/sdv-dev/CTGAN) and improves the framework in follwoing context:

- Uses the approach of firstly fitting the multivariate gaussians before training the VAE and GAN models
- Framework rewritten in PyTorch Lightning -> more flexible and easily manageble
- Implemented the Tensorboard pluggin to visualize the training processes

# Data

An example dataset will be used for the data generation pipeline. The dataset is taken from one of the most popular kaggle challanges: [Titanic](https://www.kaggle.com/competitions/titanic/data). Download and put it into the folder `data`.

Another used dataset is from the challenge [Housing Prices competition](https://www.kaggle.com/competitions/home-data-for-ml-course/data). This dataset constists of more continuous data features.

# Environment

Create environment as follows:

```
conda env create -f environment.yml
```

After the environment is installed you can activate it:

```
conda activate gen-tab-env
```
# Usage

You need to firstly setup the information in **conf/generation_config.yaml** considering the path tou your data.

Afterwards you can run a training of the variational autoencoder as follows

```
python -m scripts.train_vae
```
or use the created setup for VSCode Debug.

# Tensorboard

To run tesorboard visualizing your algorithms use Tensorboard Extension in VSCode or run 

```
python -m tensorboard.main --logdir [logdir name]
```
