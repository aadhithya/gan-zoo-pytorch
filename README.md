# pytorch-gan-zoo
A zoo of GAN implementations

## 0. Models
- [ ] DCGAN
- [ ] WGAN
- [x] WGAN-GP

## 1. Install dependencies
`pip install -r requirements.txt`

## 2. train model
- currently uses `torchvision.datasets.ImageFolder` Dataset. Change `data_dir` parameter in `config.yml` to your custom dataset path.
- only WGAN-GP supported for now.
`python train.py wgan-gp --config-dir path/to/config.yml`

## 3. `config.yml`


## To-Do
- [ ] Model Checkpointing (Save/Load).
- [ ] More Datasets
  - [ ] MNIST
  - [ ] CIFAR
- [ ] Other GANs
