# Saliency Map Interpretability as Generative Modelling

This is the reference implementation of the ICLR 2021 paper ["Rethinking the Role of Gradient-Based Attribution Methods for Model Interpretability"](https://openreview.net/forum?id=dYeAHXnpWJ4).

The repository contains methods to both train classifiers with generative score-matching regularizers, as well as methods to evaluate the interpretability and generative modelling properties of existing models.

## Usage 
First train regularized models using 

`
python train.py --regularizer=<score/anti/gnorm> --model-arch=<resnet9/18/34> --dataset=<cifar10/cifar100>
`

Next perform evaluations on trained models using

`
python evaluations.py --eval=<visualize-saliency-and-samples/compute-sample-quality/pixel-perturb> --model-name=<path-to-model> --model-arch=<resnet9/18/34> --dataset=<cifar10/cifar100>
`

Please see `example_scripts.sh` for specific examples.


## Structure 

The code is organized as follows:

- `regularizers.py` defines `ScoreMatching`, `AntiScoreMatching` and `GradNormRegularizer` methods

- `train.py` contains scripts to train CIFAR10/100 models with (and without) above regularizers

- `evaluations.py` contains scripts to perform the following evaluations on trained models:
    - `visualize-saliency-and-samples`, which computes saliency maps, and samples using activation maximization on the outputs, and dumps the resulting output images
    - `compute-sample-quality`, generate samples using activation maximization as above, and use the GAN-test to evaluate sample quality
    - `pixel-perturb`, evaluate saliency map faithfulness using the pixel perturbation method


- `example_scripts.sh` contains example scripts to run `train.py` and `evaluations.py`

- `utils\` folder contains routines to compute gradients, perform pixel perturbation, activation maximization and other miscellaneous functions

- `models\` folder contains model definitions


## Dependencies
```
torch torchvision numpy matplotlib pandas
```

## Research
If you found our work helpful for your research, please do consider citing us.
```
@inproceedings{srinivas2021rethinking,
title={Rethinking the Role of Gradient-based Attribution Methods for Model Interpretability},
author={Suraj Srinivas and Francois Fleuret},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=dYeAHXnpWJ4}
}
```