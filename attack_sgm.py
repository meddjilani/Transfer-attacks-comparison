from comet_ml import Experiment
import argparse
import torch
import torch.nn as nn
from torchattacks.attacks import pgd, mifgsm

from Skip_gradient_method.utils_sgm import register_hook_for_resnet, register_hook_for_densenet
import robustbench

import os, json
from robustbench.data import load_cifar10
from cifar10_models.resnet import resnet50

from app_config import COMET_APIKEY, COMET_WORKSPACE, COMET_PROJECT
from torch.utils.data import TensorDataset, DataLoader


parser = argparse.ArgumentParser(description='PyTorch SGM Attack Evaluation')
parser.add_argument('--model', type=str, default='Gowal2021Improving_28_10_ddpm_100m')
parser.add_argument('--target', type=str, default= 'Standard')
parser.add_argument('--n_examples', type=int, default=10)
parser.add_argument('--input_dir', default='./cifar10_32', help='the path of original dataset') #medben
parser.add_argument('--output_dir', default='adv_images', help='the path of the saved dataset')
parser.add_argument('--arch', default='resnet50',
                    help='source model for black-box attack evaluation',)
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for adversarial attack')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--eps', default=8, type=float,
                    help='perturbation')
parser.add_argument('--steps', dest='steps', default=10, type=int,
                    help='perturb number of steps')
parser.add_argument('--alpha', dest='alpha', default=2, type=float,
                    help='perturb step size')
parser.add_argument('--gamma', default=0.5, type=float)
parser.add_argument('--momentum', default=0.0, type=float)
parser.add_argument('--print_freq', default=10, type=int)
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

parameters = {'attack': 'SGM', **vars(args), **config}
experiment.log_parameters(parameters)

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}


# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]



def main():
    x_t, y_t = load_cifar10(n_examples=args.n_examples) if args.dataset == "cifar10" else None
    dataset = TensorDataset(torch.FloatTensor(x_t), torch.LongTensor(y_t))

    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                        pin_memory=True)

    target_model = robustbench.utils.load_model(model_name=args.target, dataset=args.dataset, threat_model='Linf')
    target_model.to(device)

    # create models
    if args.model != "":
        model = robustbench.utils.load_model(model_name=args.model, dataset=args.dataset, threat_model='Linf')
        args.arch = str(model.__class__).split(".")[-1].lower()[:-2]
        experiment.log_parameter("arch", args.arch)
    else:
        net = resnet50(pretrained=True)
        model = nn.Sequential(Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616]), net)  # medben
    model = model.to(device)
    model.eval()
    print()
    # create adversary attack
    epsilon = args.eps
    step_size = args.alpha

    # if args.gamma < 1.0:
    #     print('using our method')
    #     register_hook(model, args.arch, args.gamma, is_conv=args.is_conv)

    # using our method - Skip Gradient Method (SGM)
    if args.gamma < 1.0:
        if "densenet" in args.arch:
            register_hook_for_densenet(model, arch=args.arch, gamma=args.gamma)
        elif "resnet" in args.arch:
            register_hook_for_resnet(model, arch=args.arch, gamma=args.gamma)
        else:
            raise ValueError('Current code only supports resnet/densenet. '
                             'You can extend this code to other architectures.')

    if args.momentum > 0.0:
        print('using PGD attack with momentum = {}'.format(args.momentum))

        adversary = mifgsm.MIFGSM(model=model, eps=epsilon, alpha=step_size, steps=args.steps, decay=args.momentum)
        adversary.targeted = False
    else:
        print('using linf PGD attack')
        adversary = pgd.PGD(model=model, eps=epsilon, alpha=step_size, steps=args.steps, random_start=False)
        adversary.targeted = False

    for batch_ndx, (x_test, y_test) in enumerate(data_loader):

        x_test, y_test = x_test.to(device), y_test.to(device)
        acc = robustbench.utils.clean_accuracy(target_model,x_test ,y_test)
        print(args.target, 'Clean Acc:', acc)

        adv_images = adversary(x_test, y_test)
        rob_acc = robustbench.utils.clean_accuracy(target_model, adv_images, y_test)
        print(args.target, 'Robust Acc:', rob_acc)
        metrics = {'clean_acc': acc, 'robust_acc': rob_acc}
        experiment.log_metrics(metrics, step=1)

if __name__ == '__main__':
    main()
