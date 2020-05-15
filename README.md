# Introduction
This project uses GAN to generate desired curve, which implemented by keras. In [1][2], there are implementations by tensorflow and pytorch
respectively; this project uses keras. This is one thing i want to mention: in this project, i use different loss function than original paper; see loss function and reasons in [4].
# Run code
```
python GAN.py
```
# Recent update
Added Wasserstein GAN. Even the reasons why WGAN performs better than GAN are complicated, the differences of code by using keras are minor. In fact, only few lines need to be changed in this example.
## Run code
```
python WGAN.py
```
# Future work
Implement different type of GAN.
# Reference
[1][Tensorflow-Tutorial](https://github.com/MorvanZhou/Tensorflow-Tutorial.git)
[2][PyTorch-Tutorial](https://github.com/MorvanZhou/PyTorch-Tutorial.git)
[3][Keras-GAN](https://github.com/eriklindernoren/Keras-GAN.git)
[4][ganhacks](https://github.com/soumith/ganhacks.git)
