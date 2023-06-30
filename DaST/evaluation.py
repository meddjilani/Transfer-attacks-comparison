from comet_ml import Experiment
import argparse
import os
import random
import numpy as np
import torchattacks
import foolbox
import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.utils.data.sampler as sp
from torch.utils.data import TensorDataset, DataLoader
from robustbench.data import load_cifar10
from robustbench.utils import load_model, clean_accuracy

from app_config import COMET_APIKEY, COMET_WORKSPACE, COMET_PROJECT

SEED = 10000
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(10000)

parser = argparse.ArgumentParser()
parser.add_argument('--substitute_model', type=str, default='Gowal2021Improving_28_10_ddpm_100m')
parser.add_argument('--model', type=str, default='Carmon2019Unlabeled')
parser.add_argument('--n_examples', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=5)

parser.add_argument('--eps', type = float, default=8/255)
parser.add_argument('--alpha', type=float,default=2/255)
parser.add_argument('--decay', type=float,default= 1.0)
parser.add_argument('--steps', type=int,default=10)

parser.add_argument('--path', type=str, default='saved_model/netD_epoch_1.pth', help='attacked network path')
parser.add_argument('--adv', type=str, default='BIM', help='attack method')
parser.add_argument('--mode', type=str, default='dast', help='use which model to generate\
    examples. "imitation_large": the large imitation network.\
    "imitation_medium": the medium imitation network. "limitation_small" the\
    small imitation network. ')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--target', action='store_true', help='manual seed')
parser.add_argument('--target_label', type=int, default=0)


opt = parser.parse_args()
# print(opt)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
# print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

experiment = Experiment(
    api_key=COMET_APIKEY,
    project_name=COMET_PROJECT,
    workspace=COMET_WORKSPACE,
)
parameters = {'attack': 'DAST', **vars(opt)}
experiment.log_parameters(parameters)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_t, y_t = load_cifar10(n_examples=opt.n_examples)
dataset = TensorDataset(torch.FloatTensor(x_t), torch.LongTensor(y_t))
testloader = DataLoader(dataset, batch_size=opt.batch_size,
                    pin_memory=True)


def test_adver(net, tar_net, attack, target):
    net.eval()
    tar_net.eval()
    # MIFGSM
    if attack == 'MIFGSM':
        attack_method = torchattacks.MIFGSM(net, eps=opt.eps, alpha=opt.alpha, steps=opt.steps, decay=opt.decay)
    # BIM
    elif attack == 'BIM':
        attack_method = torchattacks.BIM(net, eps=opt.eps, alpha=opt.alpha, steps=opt.steps) 
    # FGSM
    elif attack == 'FGSM':
        attack_method = torchattacks.FGSM(net, eps=opt.eps)


    # ----------------------------------
    # Obtain the attack success rate of the model
    # ----------------------------------
    for data in testloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        if target:
            total = labels.size(0)
            attack_method.targeted = True
            attack_method.set_mode_targeted_by_label()
            target_labels = torch.full_like(labels, opt.target_label)
            adv_images = attack_method(inputs, target_labels)

            outputs = tar_net(adv_images)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == target_labels).sum()
            rob_acc = (correct.float() / total)
            print(opt.model, 'Robust Acc: %2.2f %%'%(100 - rob_acc*100))

        else:
            attack_method.targeted = False
            adv_images = attack_method(inputs, labels)
            rob_acc = clean_accuracy(tar_net, adv_images, labels)
            print(opt.model, 'Robust Acc: %2.2f %%'%(rob_acc*100))
            
        acc = clean_accuracy(tar_net, inputs, labels)
        print(opt.model, 'Clean Acc: %2.2f %%'%(acc*100))
        metrics = {'clean_acc': acc, 'robust_acc': rob_acc}
        experiment.log_metrics(metrics, step=1)

target_net = load_model(model_name=opt.model, dataset='cifar10', threat_model='Linf')
target_net = target_net.to(device)

if opt.mode == 'black':
    attack_net = load_model(model_name=opt.substitute_model, dataset='cifar10', threat_model='Linf')
    attack_net = attack_net.to(device)
elif opt.mode == 'white':
    attack_net = target_net
elif opt.mode == 'dast':
    attack_net = load_model(model_name=opt.substitute_model, dataset='cifar10', threat_model='Linf')
    attack_net = attack_net.to(device)
    state_dict = torch.load(opt.path)
    attack_net.load_state_dict(state_dict)
    attack_net.eval()

test_adver(attack_net, target_net, opt.adv, opt.target)