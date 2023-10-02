from comet_ml import Experiment
import os
import json
import torch
from robustbench.data import load_cifar10
from robustbench.utils import load_model, clean_accuracy
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
import torch.nn as nn
#from app_config import COMET_APIKEY, COMET_WORKSPACE, COMET_PROJECT
from utils.admix import Admix_Attacker
import argparse

from cifar10_models.vgg import vgg16_bn,vgg11_bn,vgg19_bn, vgg13_bn
from cifar10_models.resnet import resnet18,resnet34,resnet50
from cifar10_models.inception import inception_v3
from cifar10_models.googlenet import googlenet
from cifar10_models.densenet import densenet161, densenet121, densenet169
from cifar10_models.mobilenetv2 import mobilenet_v2
from Normalize import Normalize


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Gowal2021Improving_28_10_ddpm_100m')
    parser.add_argument('--target', type=str, default= 'Carmon2019Unlabeled')
    parser.add_argument('--n_examples', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--eps', default=8/255, type=float, help='perturbation')
    parser.add_argument('--nb_iter', dest='steps', default=10, type=int,help='perturb number of steps')
    parser.add_argument("--admix-m1", default=5, dest='admix_m1', type=float, help="Admix")
    parser.add_argument("--admix-m2", default=3, dest='admix_m2', type=float, help="Admix")
    parser.add_argument("--admix-portion", default=0.2, dest='admix_portion', type=float, help="Admix")
    args = parser.parse_args()

    config = {}
    if os.path.exists('config_ids_source_targets.json'):
        with open('config_ids_source_targets.json', 'r') as f:
            config = json.load(f)

    experiment = Experiment(
        api_key="RDxdOE04VJ1EPRZK7oB9a32Gx",
        project_name="compare-sota",
        workspace="meddjilani",
    )

    parameters = {'attack': 'ADMIX', **vars(args), **config}
    experiment.log_parameters(parameters)
    experiment.set_name("ADMIX_"+args.model+"_"+args.target)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2471, 0.2435, 0.2616]

    if args.model =='vgg19':
        source_model = vgg19_bn(pretrained=True)
    elif args.model =='vgg16':
        source_model = vgg16_bn(pretrained=True)
    elif args.model =='vgg13':
        source_model = vgg13_bn(pretrained=True)
    elif args.model =='vgg11':
        source_model = vgg11_bn(pretrained=True)
    elif args.model =='resnet18':
        source_model = resnet18(pretrained=True)
    elif args.model =='resnet34':
        source_model = resnet34(pretrained=True)
    elif args.model =='resnet50':
        source_model = resnet50(pretrained=True)
    elif args.model =='densenet121':
        source_model = densenet121(pretrained=True)
    elif args.model =='densenet161':
        source_model = densenet161(pretrained=True)
    elif args.model =='densenet169':
        source_model = densenet169(pretrained=True)
    elif args.model =='googlenet':
        source_model = googlenet(pretrained=True)
    elif args.model =='mobilenet':
        source_model = mobilenet(pretrained=True)
    elif args.model =='inception':
        source_model = inception_v3(pretrained=True)

    if args.target =='vgg19':
        target_model = vgg19_bn(pretrained=True)
    elif args.target =='vgg16':
        target_model = vgg16_bn(pretrained=True)
    elif args.target =='vgg13':
        target_model = vgg13_bn(pretrained=True)
    elif args.target =='vgg11':
        target_model = vgg11_bn(pretrained=True)
    elif args.target =='inception':
        target_model = inception_v3(pretrained=True)
    elif args.target =='googlenet':
        target_model = googlenet(pretrained=True)
    elif args.target =='mobilenet':
        target_model = mobilenet_v2(pretrained=True)
    elif args.target =='resnet50':
        target_model = resnet50(pretrained=True)
    elif args.target =='resnet34':
        target_model = resnet34(pretrained=True)
    elif args.target =='resnet18':
        target_model = resnet18(pretrained=True)
    elif args.target =='densenet169':
        target_model = densenet169(pretrained=True)
    elif args.target =='densenet161':
        target_model = densenet161(pretrained=True)
    elif args.target =='densenet121':
        target_model = densenet121(pretrained=True)
    elif args.target =='mobilenet_v2':
        target_model = mobilenet_v2(pretrained=True)


    source_model = nn.Sequential(
        Normalize(mean, std),
        source_model
    )
    source_model.eval() 
    source_model.to(device)

    target_model = nn.Sequential(
        Normalize(mean, std),
        target_model
    )
    target_model.eval() 
    target_model.to(device)

    x_test, y_test = load_cifar10(n_examples=args.n_examples)
    x_test, y_test = x_test.to(device), y_test.to(device)

    dataset_test = TensorDataset(x_test, y_test)
    data_loader = DataLoader(dataset_test, batch_size=args.batch_size)

    #admix
    attack = Admix_Attacker("admix",source_model, F.cross_entropy,args)

    i = 0
    for batch_ndx, (inputs, labels) in enumerate(data_loader):
        print('\nRunning Admix attack')

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            _, preds = torch.max(source_model(inputs), dim=1)

        inputs_adv = attack.perturb(inputs, preds)

        acc = clean_accuracy(target_model, inputs, labels) 
        rob_acc = clean_accuracy(target_model, inputs_adv, labels) 
        print(args.target, 'Clean Acc: %2.2f %%'%(acc*100))
        print(args.target, 'Robust Acc: %2.2f %%'%(rob_acc*100))

        with torch.no_grad():
            predictions = target_model(inputs)
            predicted_classes = torch.argmax(predictions, dim=1)
            correct_predictions = (predicted_classes == labels).sum().item()
            correct_batch_indices = (predicted_classes == labels).nonzero().squeeze(-1)
        
        suc_rate = 1 - clean_accuracy(target_model, inputs_adv[correct_batch_indices,:,:,:], labels[correct_batch_indices])
        print(args.target, 'Success Rate: %2.2f %%'%(suc_rate*100))
        metrics = {'clean_acc': acc, 'robust_acc': rob_acc, 'suc_rate': suc_rate, 'target_correct_pred': correct_predictions}
        experiment.log_metrics(metrics, step=batch_ndx)


        # if i==0:
        #     adv_images_Admix = inputs_adv
        # else:
        #     adv_images_Admix = torch.cat([adv_images_Admix,inputs_adv],dim=0)
        # i += 1

