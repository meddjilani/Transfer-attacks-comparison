from comet_ml import Experiment
import torch
import torch.nn as nn
from cifar10_models.resnet import *
from cifar10_models.densenet import *
from cifar10_models.vgg import *
from robustbench.data import load_cifar10
from robustbench.utils import load_model, clean_accuracy
import torchattacks_ens.attacks.mifgsm_ens as taamifgsm_ens
import json
import os
from Normalize import Normalize
import argparse
from comet_ml import Experiment


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, nargs = '+',default=['Gowal2021Improving_28_10_ddpm_100m', 'Standard'])
    parser.add_argument('--target', type=str, default= 'Carmon2019Unlabeled')
    parser.add_argument('--n_examples', type=int, default=10)
    parser.add_argument('--eps', type = float, default=8/255)
    parser.add_argument('--alpha', type=float,default=2/255)
    parser.add_argument('--decay', type=float,default= 1.0)
    parser.add_argument('--steps', type=int,default=10)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    source_models = []
    for model_name in args.model:
        source_model = load_model(model_name=model_name, dataset='cifar10', threat_model='Linf')
        source_model.to(device)
        source_models.append(source_model)


    target_model = load_model(model_name= args.target, dataset='cifar10', threat_model='Linf')
    target_model.to(device)


    x_test, y_test = load_cifar10(n_examples=args.n_examples)
    x_test, y_test = x_test.to(device), y_test.to(device)

    print('Running MI-FGSM ENS attack')
    attack = taamifgsm_ens.MIFGSM_ENS(source_models[0], source_models, eps=args.eps, alpha=args.alpha, steps=args.steps, decay=args.decay)
    adv_images_MI_ENS = attack(x_test, y_test)


    acc = clean_accuracy(target_model, x_test, y_test) 
    rob_acc = clean_accuracy(target_model, adv_images_MI_ENS, y_test) 
    print(args.target, 'Clean Acc: %2.2f %%'%(acc*100))
    print(args.target, 'Robust Acc: %2.2f %%'%(rob_acc*100))

    experiment = Experiment(
    api_key="RDxdOE04VJ1EPRZK7oB9a32Gx",
    project_name="Black-box attack comparison cifar10",
    workspace="meddjilani",
    )

    metrics = {'Clean accuracy': acc, 'Robust accuracy': rob_acc}
    experiment.log_metrics(metrics, step=1)

    parameters = {'attack':'MI-FGSM ENS', 'sources':args.model, 'target':args.target, 'n_examples':args.n_examples, 'eps':args.eps, 'alpha':args.alpha, 'steps':args.steps, 'decay':args.decay}
    experiment.log_parameters(parameters)