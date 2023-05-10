from comet_ml import Experiment
import os
import json
import torch
from robustbench.data import load_cifar10
from robustbench.utils import load_model, clean_accuracy
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
from app_config import COMET_APIKEY, COMET_WORKSPACE, COMET_PROJECT

from admix import Admix_Attacker
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Gowal2021Improving_28_10_ddpm_100m')
    parser.add_argument('--target', type=str, default= 'Carmon2019Unlabeled')
    parser.add_argument('--n_examples', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--admix-m1", dest='admix_m1', type=float, help="Admix")
    parser.add_argument("--admix-m2", dest='admix_m2', type=float, help="Admix")
    parser.add_argument("--admix-portion", dest='admix_portion', type=float, help="Admix")
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

    parameters = {'attack': 'ADMIX', **vars(args), **config}
    experiment.log_parameters(parameters)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    source_model = load_model(model_name=args.model, dataset='cifar10', threat_model='Linf')
    source_model.to(device)

    target_model = load_model(model_name= args.target, dataset='cifar10', threat_model='Linf')
    target_model.to(device)

    x_test, y_test = load_cifar10(n_examples=args.n_examples)
    x_test, y_test = x_test.to(device), y_test.to(device)

    dataset_test = TensorDataset(x_test, y_test)
    data_loader = DataLoader(dataset_test, batch_size=args.batch_size)

    #admix
    attack = Admix_Attacker("admix",source_model, F.cross_entropy,args)

    source_model.eval()
    i = 0
    for inputs, labels in data_loader:
        inputs = inputs#.to(device)
        labels = labels#.to(device)

        with torch.no_grad():
            _, preds = torch.max(source_model(inputs), dim=1)

        inputs_adv = attack.perturb(inputs, preds)
        if i==0:
            adv_images_Admix = inputs_adv
        else:
            adv_images_Admix = torch.cat([adv_images_Admix,inputs_adv],dim=0)
        i += 1

    #admix

    acc = clean_accuracy(target_model, x_test, y_test) 
    rob_acc = clean_accuracy(target_model, adv_images_Admix, y_test) 
    print(args.target, 'Clean Acc: %2.2f %%'%(acc*100))
    print(args.target, 'Robust Acc: %2.2f %%'%(rob_acc*100))


    metrics = {'clean_acc': acc, 'robust_acc': rob_acc}
    experiment.log_metrics(metrics, step=1)
