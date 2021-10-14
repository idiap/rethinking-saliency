## Scripts to train models

# train a baseline cifar model
python train.py --model-arch=resnet18 --dataset=cifar100 

# train a cifar model with score matching
python train.py --model-arch=resnet18 --dataset=cifar100 --regularizer=score --regularization-constant=1e-5

# train a cifar model with gradient norm regularization
python train.py --model-arch=resnet18 --dataset=cifar100 --regularizer=gnorm --regularization-constant=1e-5

# train a cifar model with anti score matching
python train.py --model-arch=resnet18 --dataset=cifar100 --regularizer=anti --regularization-constant=1e-5

## Scripts to evaluate

# dump saliency maps and activation maximization samples
python evaluations.py --model-name='saved_models/resnet18_cifar100.pt' --model-arch=resnet18 --dataset=cifar100 --eval=visualize-saliency-and-samples

# evaluate sample quality using SGLD
python evaluations.py --model-name='saved_models/resnet18_cifar100.pt' --model-arch=resnet18 --dataset=cifar100 --eval=compute-sample-quality

# perform pixel perturbation test
python evaluations.py --model-name='saved_models/resnet18_cifar100.pt' --model-arch=resnet18 --dataset=cifar100 --eval=pixel-perturb

## Script to plot results saved (for example) in 'results/pixel-perturb' folder

python plot_results.py --csv_path='results/pixel-perturb/'
