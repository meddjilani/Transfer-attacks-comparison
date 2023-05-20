from comet_ml import Experiment
import torch
from robustbench.data import load_cifar10
from robustbench.utils import load_model, clean_accuracy
import torchattacks
import json
import os
import argparse
from app_config import COMET_APIKEY, COMET_WORKSPACE, COMET_PROJECT
from torch.utils.data import TensorDataset, DataLoader

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Gowal2021Improving_28_10_ddpm_100m')
    parser.add_argument('--target', type=str, default= 'Carmon2019Unlabeled')
    parser.add_argument('--n_examples', type=int, default=10)
    parser.add_argument('--eps', type = float, default=8/255)
    parser.add_argument('--alpha', type=float,default=2/255)
    parser.add_argument('--decay', type=float,default= 1.0)
    parser.add_argument('--steps', type=int,default=10)  
    parser.add_argument('--resize_rate', type=float,default= 0.9)
    parser.add_argument('--diversity_prob', type=float,default= 0.5)
    parser.add_argument('--random_start', type=bool,default= False) 
    parser.add_argument('--kernel_name', type=str,default= 'gaussian')
    parser.add_argument('--len_kernel', type=int,default= 15)
    parser.add_argument('--nsig', type=int,default= 3)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "imagenet"], default="cifar10")

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

    parameters = {'attack': 'TI-FGSM', **vars(args), **config}
    experiment.log_parameters(parameters)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    source_model = load_model(model_name=args.model, dataset='cifar10', threat_model='Linf')
    source_model.to(device)

    target_model = load_model(model_name= args.target, dataset='cifar10', threat_model='Linf')
    target_model.to(device)

    x_t, y_t = load_cifar10(n_examples=args.n_examples) if args.dataset == "cifar10" else None
    dataset = TensorDataset(torch.FloatTensor(x_t), torch.LongTensor(y_t))

    loader = DataLoader(dataset, batch_size=args.batch_size,
                        pin_memory=True)

    for batch_ndx, (x_test, y_test) in enumerate(loader):
        x_test, y_test = x_test.to(device), y_test.to(device)

        print('Running TI-FGSM attack')
        attack = torchattacks.TIFGSM(source_model, eps=args.eps, alpha=args.alpha, steps=args.steps, decay=args.decay,
                                    resize_rate=args.resize_rate, diversity_prob=args.diversity_prob, random_start=args.random_start,
                                    kernel_name=args.kernel_name, len_kernel=args.len_kernel, nsig=args.nsig)
        adv_images_TI = attack(x_test, y_test)

        acc = clean_accuracy(target_model, x_test, y_test)
        rob_acc = clean_accuracy(target_model, adv_images_TI, y_test)
        print(args.target, 'Clean Acc: %2.2f %%'%(acc*100))
        print(args.target, 'Robust Acc: %2.2f %%'%(rob_acc*100))

        metrics = {'clean_acc': acc, 'robust_acc': rob_acc}
        experiment.log_metrics(metrics, step=1)

