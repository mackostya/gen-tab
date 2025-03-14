# Tabular data generation
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)
<!-- Images in a row -->
<p align="center">
  <img src="imgs/val_loss.png" alt="Image1" width="208" />
  <img src="imgs/weights.png" alt="Image2" width="200" />
  <img src="imgs/grads.png" alt="Image3" width="204" />
  <img src="imgs/feature_generation_evolution.gif" alt="Image4" width="204" />
</p>

This project serves as an example ML / DL project for tabular data generation. It takes at the basis the approach of
[CTGAN and TVAE](https://github.com/sdv-dev/CTGAN) and improves the framework in follwoing context:

- The data transformer is saved and reused for the data, if not requested differently
- Framework rewritten in PyTorch Lightning -> more flexible and easily manageble
- Implemented the Tensorboard pluggin to visualize the training processes

# Data

An example dataset will be used for the data generation pipeline. The dataset is taken from one of the most popular kaggle challanges: [Housing Prices competition](https://www.kaggle.com/competitions/home-data-for-ml-course/data). Download and put it into the folder `data`.

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
python -m scripts.train --model-type=[model]
```

while acceptable `[model]` is either `vae` (default) or `gan`. Or use the created setup for VSCode Debug (see the image below).
<p align="left">
  <img src="imgs/run_with_launch_json.png" alt="Image1" width="208" />
</p>

# Tensorboard


To run tesorboard visualizing your algorithms use Tensorboard Extension in VSCode or run 

```
python -m tensorboard.main --logdir [logdir name]  --samples_per_plugin "images=100"
```

If you want to specialize specific blocks of your architecture, including their histomgrams, weights and gradients please edit them in [tb_settings.yaml](https://github.com/mackostya/gen-tab/blob/main/conf/tb_settings.yaml). Add them to `parameters_to_visualize`.