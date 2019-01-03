## Nostalgic Adam
### repository for the paper: Nostalgic Adam: Weighing more of the past gradients when designing the adaptive learning rate
Haiwen Huang, Chang Wang, Bin Dong (https://arxiv.org/abs/1805.07557) 
The code is implemented using Pytorch0.4

To run the training process, download the repository, and use 
''python train_cifar.py --optimizer nosadam --epoch 200 --lr 0.01''


1. two main optimizers are adastab.py and OurAdam.py, with the former being NosAdam optimizer. (AdaStab is its original name) And they can be used just like any other optimizers in your own code, see examples in the iPython Notebooks. Also note that AMSGrad can also be implemented using OurAdam.py, by setting the argument AMSGrad=True.

2. the iPython Notebooks are the training logs of the experiments in the paper (Section 5).
