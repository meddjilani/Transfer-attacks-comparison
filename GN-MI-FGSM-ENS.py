from comet_ml import Experiment
import torch
import torch.nn as nn
from robustbench.data import load_cifar10
from robustbench.utils import load_model, clean_accuracy
import torchattacks_ens.attacks.mifgsm_ens as taamifgsm_ens
import json
import os
import argparse
from app_config import COMET_APIKEY, COMET_WORKSPACE, COMET_PROJECT
from torch.utils.data import TensorDataset, DataLoader

from utils.modelzoo_ghost.robustbench.robustbench.utils import load_model_ghost
from cifar10_models.vgg import vgg16_bn,vgg11_bn,vgg19_bn, vgg13_bn
from cifar10_models.resnet import resnet18,resnet34,resnet50
from cifar10_models.inception import inception_v3
from cifar10_models.googlenet import googlenet
from cifar10_models.densenet import densenet161, densenet121, densenet169
from cifar10_models.mobilenetv2 import mobilenet_v2

from cifar10_models.resnet_ghost import resnet18 as resnet18_gn, resnet34 as resnet34_gn, resnet50 as resnet50_gn
from cifar10_models.densenet_ghost import densenet161 as densenet161_gn, densenet121 as densenet121_gn, densenet169 as densenet169_gn
from Normalize import Normalize
from utils import set_random_seed


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', nargs='+', default=['Standard','Ding2020MMA'])
    parser.add_argument('--target', type=str, default= 'Carmon2019Unlabeled')
    parser.add_argument('--target_label', type=int, default=2)
    parser.add_argument("--targeted", action='store_true', help="run targeted attack")
    parser.add_argument('--n_im', type=int, default=10)
    parser.add_argument('--eps', type = float, default=8/255)
    parser.add_argument('--alpha', type=float,default=2/255)
    parser.add_argument('--decay', type=float,default= 1.0)
    parser.add_argument('--steps', type=int,default=10)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "imagenet"], default="cifar10")
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    set_random_seed(args.seed)

    config = {}
    if os.path.exists('config_ids_source_targets.json'):
        with open('config_ids_source_targets.json','r') as f:
            config = json.load(f)

    experiment = Experiment(
        api_key=COMET_APIKEY,
        project_name=COMET_PROJECT,
        workspace=COMET_WORKSPACE,
    )
    parameters = {'attack': 'GN-MI-FGSM-ENS', **vars(args), **config}
    experiment.log_parameters(parameters)
    experiment.set_name("GN-MI-FGSM-ENS_"+"_".join(args.model)+"_"+args.target)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2471, 0.2435, 0.2616]
    source_models = []
    for model_name in args.model:
        robust_model_source = False
        if model_name =='vgg19':
            source_model = vgg19_bn(pretrained=True)
        elif model_name =='vgg16':
            source_model = vgg16_bn(pretrained=True)
        elif model_name =='vgg13':
            source_model = vgg13_bn(pretrained=True)
        elif model_name =='vgg11':
            source_model = vgg11_bn(pretrained=True)
        elif model_name =='resnet18':
            source_model = resnet18_gn(pretrained=True)
        elif model_name =='resnet34':
            source_model = resnet34_gn(pretrained=True)
        elif model_name =='resnet50':
            source_model = resnet50_gn(pretrained=True)
        elif model_name =='densenet121':
            source_model = densenet121_gn(pretrained=True)
        elif model_name =='densenet161':
            source_model = densenet161_gn(pretrained=True)
        elif model_name =='densenet169':
            source_model = densenet169_gn(pretrained=True)
        elif model_name =='googlenet':
            source_model = googlenet(pretrained=True)
        elif model_name =='mobilenet':
            source_model = mobilenet(pretrained=True)
        elif model_name =='inception':
            source_model = inception_v3(pretrained=True)
        else:
            robust_model_source = True
        if robust_model_source:
            source_model = load_model_ghost(model_name, dataset='cifar10', threat_model='Linf')
        else:
            source_model = nn.Sequential(
                Normalize(mean, std),
                source_model
            )
            source_model.eval()
        source_model.to(device)
        source_models.append(source_model)

    robust_model_target = False
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
    else:
        robust_model_target = True

    if robust_model_target:
        target_model = load_model(args.target, dataset='cifar10', threat_model='Linf')
    else:
        target_model = nn.Sequential(
            Normalize(mean, std),
            target_model
        )
        target_model.eval() 
    target_model.to(device)



    x_t, y_t = load_cifar10(args.n_im) if args.dataset=="cifar10" else None
    dataset = TensorDataset(torch.FloatTensor(x_t), torch.LongTensor(y_t))

    loader = DataLoader(dataset, batch_size=args.batch_size,
                        pin_memory=True)
    suc_rate_steps = 0
    images_steps = 0
    for batch_ndx, (x_test, y_test) in enumerate(loader):

        x_test, y_test = x_test.to(device), y_test.to(device)

        print('Running GN-MI-FGSM ENS attack')
        attack = taamifgsm_ens.MIFGSM_ENS(source_models[0], source_models, eps=args.eps, alpha=args.alpha, steps=args.steps, decay=args.decay)
        if args.targeted:
            attack.targeted = True
            attack.set_mode_targeted_by_label()
            target_labels = torch.full_like(y_test, args.target_label)
            adv_images_GN_MI_ENS = attack(x_test, target_labels)
        else:
            attack.targeted = False
            attack = taamifgsm_ens.MIFGSM_ENS(source_models[0], source_models, eps=args.eps, alpha=args.alpha, steps=args.steps, decay=args.decay)
            adv_images_GN_MI_ENS = attack(x_test, y_test)


        acc = clean_accuracy(target_model, x_test, y_test)
        rob_acc = clean_accuracy(target_model, adv_images_GN_MI_ENS, y_test)
        print(args.target, 'Clean Acc: %2.2f %%'%(acc*100))
        print(args.target, 'Robust Acc: %2.2f %%'%(rob_acc*100))

        with torch.no_grad():
            predictions = target_model(x_test)
            predicted_classes = torch.argmax(predictions, dim=1)
            correct_predictions = (predicted_classes == y_test).sum().item()
            correct_batch_indices = (predicted_classes == y_test).nonzero().squeeze(-1)
        
        suc_rate = 1 - clean_accuracy(target_model, adv_images_GN_MI_ENS[correct_batch_indices,:,:,:], y_test[correct_batch_indices])
        print(args.target, 'Success Rate: %2.2f %%'%(suc_rate*100))
        if correct_batch_indices.size(0) != 0:
            suc_rate_steps = suc_rate_steps*images_steps + suc_rate*correct_batch_indices.size(0)
            images_steps += correct_batch_indices.size(0)
            suc_rate_steps = suc_rate_steps/images_steps
        metrics = {'suc_rate_steps':suc_rate_steps, 'clean_acc': acc, 'robust_acc': rob_acc, 'suc_rate': suc_rate, 'target_correct_pred': correct_predictions}
        experiment.log_metrics(metrics, step=batch_ndx+1)