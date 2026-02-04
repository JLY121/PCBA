# PCBA: A Multi-Stage Backdoor Attack Framework for the Pre-Convergence Phase of Federated Learning

### Create Environment

```
conda create -n your_env python=3.10.14
conda activate your_env
pip install -r requirement.txt
```

### Quick start
```
cd main

python clean.py --gpu 0 --params configs/Our_cifar10_resnet18.yaml                  # for pretrain

python clean.py --gpu 0 --params configs/pretrain_cifar10_resnet18.yaml             # for attack
```

### Other instructions

The code utilizes ``Weights & Biases (wandb)`` for parameter management. You can toggle wandb visualization by setting the mode to ``online\offline`` in ``main/clean.py``. 

Additionally, local visualization for the training process is implemented, with results saved by default to: ``saved/train_process_visualization``.

### Thanks

This repository is built upon the following open-source projects. We are grateful for their excellent work:
* [A3FL: Adversarially Adaptive Backdoor Attacks to Federated Learning](https://github.com/qzzqzzb/A3FL).