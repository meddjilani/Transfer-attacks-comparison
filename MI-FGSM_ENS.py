from comet_ml import Experiment
import torch
from robustbench.data import load_cifar10
from robustbench.utils import load_model, clean_accuracy
import torchattacks_ens.attacks.mifgsm_ens as taamifgsm_ens
import json
import os
import argparse

from app_config import COMET_APIKEY, COMET_WORKSPACE, COMET_PROJECT


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

    config = {}
    if os.path.exists('config_ids_source_targets.json'):
        with open('config_ids_source_targets.json', 'r') as f:
            config = json.load(f)

    experiment = Experiment(
        api_key=COMET_APIKEY,
        project_name=COMET_PROJECT,
        workspace=COMET_WORKSPACE,
    )

    parameters = {'attack': 'MI-FGSM-ENS', **vars(args), **config}
    experiment.log_parameters(parameters)

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

    metrics = {'clean_acc': acc, 'robust_acc': rob_acc}
    experiment.log_metrics(metrics, step=1)
