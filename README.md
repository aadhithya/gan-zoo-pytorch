# gan-zoo-pytorch

<img src="zoo.png" width=200></img></br>



A zoo of GAN implementations.

## 0. Models
- [ ] DCGAN
- [ ] WGAN
- [x] [WGAN-GP](https://arxiv.org/pdf/1704.00028.pdf)
## 1. Install dependencies
`pip install -r requirements.txt`

## 2. Train Model
- currently uses `torchvision.datasets.ImageFolder` Dataset. Change `data_dir` parameter in `config.yml` to your custom dataset path.
- Only 64x64 images supported.
- only WGAN-GP supported for now.

  `python train.py wgan-gp --config-dir path/to/config.yml`

## 3. `config.yml`
- Config file controls the model behaviour.
- Can be extended to have more fields as required by the model.

```
name: <str> model/config name
device: <str> [cuda|cpu] device to load models to.
data_dir: <str> path to data dir.

seed: <int> seed to control randomness.
z_dim: <int> latent dimension for generator noise input.

imsize: <int> input/output image size.
img_ch: <int> number of channels in image.

w_gp: <number> Gradient Penalty weight.
n_critic: <int> number of critic iterations.

batch_size: <int> batch size for training.
epochs: <int> number of epochs to train for.
viz_freq: <int> image vizualisation frequency (in steps).
lr:
  g: <float> learning rate for generator.
  d: <float> learning rate for discriminator/critic.
```


## To-Do
- [ ] Model Checkpointing (Save/Load).
- [ ] Flexible Image Sizes.
- [ ] Other GANs
- [ ] More Datasets
  - [ ] MNIST
  - [ ] CIFAR
