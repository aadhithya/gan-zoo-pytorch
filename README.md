# gan-zoo-pytorch

<img src="zoo.png" width=200></img></br>



A zoo of GAN implementations.

## 0. Models
- [x] [DCGAN](https://arxiv.org/pdf/1511.06434.pdf)
- [x] [WGAN](https://arxiv.org/pdf/1701.07875.pdf)
- [x] [WGAN-GP](https://arxiv.org/pdf/1704.00028.pdf)
## 1. Install dependencies
`pip install -r requirements.txt`

## 2. Train Model
### 2.1 Datasets
- currently uses `torchvision.datasets.ImageFolder` Dataset. Change `data_dir` parameter in `config.yml` to your custom dataset path.
- Only 64x64 images supported.

### 2.2 `train.py`
- to start model training, run

  `python train.py <model-name> --config-dir path/to/config.yml`
- supported models:
  1. `wgan`: Wasserstein GAN with gradient clipping.
  2. `wgan-gp`: WGAN with gradient penalty.
  3. `dcgan`: DCGAN
- for help, run

  `python train.py --help`
- for model specific help, run

  `python train.py <model_name> --help`


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

## 4. Sample Outpouts

**NOTE:** Models not trained to convergence!!

- DCGAN

![Screenshot 2021-09-01 at 18 36 33](https://user-images.githubusercontent.com/6749212/131802724-ae1489bd-5e4b-483e-8473-bf4cbf873192.png)

- WGAN

![Screenshot 2021-08-31 at 21 54 32](https://user-images.githubusercontent.com/6749212/131802784-0192eea8-592e-4a9c-b89c-1f45651f41e7.png)

- WGAN-GP

![Screenshot 2021-08-31 at 20 53 51](https://user-images.githubusercontent.com/6749212/131802831-b8baf889-0c1a-4964-9a4c-08878908bac9.png)




## To-Do
- [ ] Model Checkpointing (Save/Load).
- [ ] Flexible Image Sizes.
- [ ] Other GANs
- [ ] More Datasets
  - [ ] MNIST
  - [ ] CIFAR
