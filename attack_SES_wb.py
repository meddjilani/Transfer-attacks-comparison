"""
SES (ours)
"""

from comet_ml import Experiment
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')

from tqdm import tqdm
from utils_attacks import get_adv_np, get_label_loss, ebad_surr_losses
from robustbench.utils import load_model
import torch, json, os
import torch.nn as nn
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
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
from app_config import COMET_APIKEY, COMET_WORKSPACE, COMET_PROJECT
import random


def softmax(vector):
    exp_vector = np.exp(vector)
    normalized_vector = exp_vector / np.sum(exp_vector)
    return normalized_vector

def main():
    parser = argparse.ArgumentParser(description="BASES attack")

    parser.add_argument('--target', default='Carmon2019Unlabeled', type=str, help='Target model to use.')
    parser.add_argument('--surrogate_names',  nargs='+', default=['Standard','Ding2020MMA'], help='Surrogate models to use.')
    # parser.add_argument('--target_label',  type=int, default=2)
    parser.add_argument("--bound", default='linf', choices=['linf','l2'], help="bound in linf or l2 norm ball")
    parser.add_argument("--eps", type=int, default=8/255, help="perturbation bound: 10 for linf, 3128 for l2")
    parser.add_argument("--iters", type=int, default=10, help="number of inner iterations: 5,6,10,20...")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID: 0,1")
    parser.add_argument("--root", nargs="?", default='result', help="the folder name of result")
    
    parser.add_argument("--fuse", nargs="?", default='loss', help="the fuse method. loss or logit")
    parser.add_argument("--loss_name", nargs="?", default='cw', help="the name of the loss")
    parser.add_argument("--algo", default='mim', help="the algo used inside the perturbation machine")
    parser.add_argument("--x", type=int, default=3, help="times alpha by x")
    parser.add_argument("--lr", type=float, default=0.5, help="learning rate of w")
    parser.add_argument("--resize_rate", type=float, default=0.9, help="resize factor used in input diversity")
    parser.add_argument("--diversity_prob", type=float, default=0.5, help="the probability of applying input diversity")
    parser.add_argument("--iterw", type=int, default=50, help="iterations of updating w")
    parser.add_argument("--n_im", type=int, default=10000, help="number of images")
    parser.add_argument("--untargeted", action='store_true', help="run untargeted attack")
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
    experiment.set_name("sm_baseline_ghost_"+"_".join(args.surrogate_names)+"_"+args.target) 

    parameters = {'attack': 'QueryEnsemble', **vars(args), **config}
    experiment.log_parameters(parameters)


    algo = args.algo
    bound = args.bound
    eps = args.eps
    n_iters = args.iters
    alpha = args.x * eps / n_iters # step-size
    resize_rate = args.resize_rate
    diversity_prob = args.diversity_prob
    fuse = args.fuse
    loss_name = args.loss_name
    lr_w = float(args.lr)
    device = f'cuda:{args.gpu}'
    keys = ['w'+str(i) for i in range(args.n_im)]
    surrogate_names = args.surrogate_names

    wb = []
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2471, 0.2435, 0.2616]
    for surrogate_model in surrogate_names:
        robust_model_source = False
        if surrogate_model =='vgg19':
            source_model = vgg19_bn(pretrained=True)
        elif surrogate_model =='vgg16':
            source_model = vgg16_bn(pretrained=True)
        elif surrogate_model =='vgg13':
            source_model = vgg13_bn(pretrained=True)
        elif surrogate_model =='vgg11':
            source_model = vgg11_bn(pretrained=True)
        elif surrogate_model =='resnet18':
            source_model = resnet18_gn(pretrained=True)
        elif surrogate_model =='resnet34':
            source_model = resnet34_gn(pretrained=True)
        elif surrogate_model =='resnet50':
            source_model = resnet50_gn(pretrained=True)
        elif surrogate_model =='densenet121':
            source_model = densenet121_gn(pretrained=True)
        elif surrogate_model =='densenet161':
            source_model = densenet161_gn(pretrained=True)
        elif surrogate_model =='densenet169':
            source_model = densenet169_gn(pretrained=True)
        elif surrogate_model =='googlenet':
            source_model = googlenet(pretrained=True)
        elif surrogate_model =='mobilenet':
            source_model = mobilenet_v2(pretrained=True)
        elif surrogate_model =='inception':
            source_model = inception_v3(pretrained=True)
        else:
            robust_model_source = True
        
        if robust_model_source:
            source_model = load_model_ghost(surrogate_model, dataset='cifar10', threat_model='Linf')
        else:
            source_model = nn.Sequential(
                Normalize(mean, std),
                source_model
            )
            source_model.eval()
    
        source_model.to(device)
        wb.append(source_model)

    n_wb = len(wb)

    # load victim model
    robust_model_target = False
    if args.target =='vgg19':
        victim_model = vgg19_bn(pretrained=True)
    elif args.target =='vgg16':
        victim_model = vgg16_bn(pretrained=True)
    elif args.target =='vgg13':
        victim_model = vgg13_bn(pretrained=True)
    elif args.target =='vgg11':
        victim_model = vgg11_bn(pretrained=True)
    elif args.target =='inception':
        victim_model = inception_v3(pretrained=True)
    elif args.target =='googlenet':
        victim_model = googlenet(pretrained=True)
    elif args.target =='mobilenet':
        victim_model = mobilenet_v2(pretrained=True)
    elif args.target =='resnet50':
        victim_model = resnet50(pretrained=True)
    elif args.target =='resnet34':
        victim_model = resnet34(pretrained=True)
    elif args.target =='resnet18':
        victim_model = resnet18(pretrained=True)
    elif args.target =='densenet169':
        victim_model = densenet169(pretrained=True)
    elif args.target =='densenet161':
        victim_model = densenet161(pretrained=True)
    elif args.target =='densenet121':
        victim_model = densenet121(pretrained=True)
    else:
        robust_model_target = True

    if robust_model_target:
        victim_model = load_model(args.target, dataset='cifar10', threat_model='Linf')
    else:
        victim_model = nn.Sequential(
            Normalize(mean, std),
            victim_model
        )
        victim_model.eval() 

    victim_model.to(device)



    #load_data
    transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    ])

    testset = CIFAR10(root='/data', train = False, download = True, transform = transform_test)
    # if not args.untargeted:
    #     print('Targeted attack : Removing Label ', args.target_label,' images from Cifar10 test dataset')
    #     indices = [i for i in range(len(testset)) if testset[i][1] != args.target_label]
    #     selected_indices = indices[:args.n_im]
    # else:
    #     selected_indices = range(args.n_im)
    testloader = torch.utils.data.DataLoader( torch.utils.data.Subset(testset, range(args.n_im)), batch_size=1, shuffle = False) #sampler=torch.utils.data.sampler.SubsetRandomSampler(range(args.n_im))

    success_idx_list = set()
    success_idx_list_pretend = set() # untargeted success
    correct_pred = 0
    suc_adv = 0
    query_list = []
    query_list_pretend = []
    mean_weights = lambda tab: [sum(col) / len(tab) for col in zip(*tab)] if len(tab) > 0 else []
    w_list = []
    for im_idx,(image,label) in tqdm(enumerate(testloader)):
        loss_target = []
        print(f"im_idx: {im_idx + 1}")
        image, label = image.to(device), label.to(device)
        with torch.no_grad():
            pred_vic = torch.argmax(victim_model(image)).item()
        if label.item() == pred_vic:
            correct_pred +=1
        lr_w = float(args.lr) # re-initialize
        image = torch.squeeze(image)
        im_np = np.array(image.cpu())
        print('image mean, 1st value: ',im_np.mean(),', ',np.ravel(im_np)[0])
        gt_label = label.item()
        tgts_label = list(range(10))
        tgts_label.remove(gt_label)
        tgt_label = random.choice(tgts_label)
        if args.untargeted:
            tgt_label = gt_label
        else:
            print("tgt_label:",tgt_label)

        # weight balancing
        losses_w_b = ebad_surr_losses(im_np, tgt_label, wb)
        losses_w_b_np = np.array(losses_w_b)
        if args.untargeted:
            w_inv = losses_w_b_np
            w_np = w_inv / w_inv.sum()
            print(losses_w_b_np, w_np)            
        else:
            w_inv = 1 / losses_w_b_np
            w_np = w_inv / w_inv.sum()
            print(losses_w_b_np, w_np)


        adv_np, losses = get_adv_np(im_np, tgt_label, w_np, wb, bound, eps, n_iters, alpha, resize_rate, diversity_prob, algo = algo, fuse=fuse, untargeted=args.untargeted, loss_name=loss_name, adv_init=None)
        label_idx, loss, _ = get_label_loss(adv_np, victim_model, tgt_label, loss_name, targeted = not args.untargeted, device=device)
        n_query = 1
        loss_target.append(loss)

        # pretend
        if not args.untargeted and label_idx != gt_label:
            success_idx_list_pretend.add(im_idx)
            query_list_pretend.append(n_query)
        if (not args.untargeted and label_idx == tgt_label) or (args.untargeted and label_idx != tgt_label):
            # originally successful
            print('success', gt_label,' ------> ', label_idx,'\n')
            success_idx_list.add(im_idx)
            query_list.append(n_query)
            if label.item() == pred_vic:
                suc_adv +=1
                
        else: 
            idx_w = 0         # idx of wb in W
            last_idx = -1    # if no changes after one round, reduce the learning rate
            
            while n_query < args.iterw:
                w_np_temp_plus = w_np.copy()
                w_np_temp_plus[idx_w] += lr_w * w_inv[idx_w]
                w_np_temp_plus = softmax(w_np_temp_plus)
                adv_np_plus, losses_plus = get_adv_np(im_np, tgt_label, w_np_temp_plus, wb, bound, eps, n_iters, alpha, resize_rate, diversity_prob, algo = algo, fuse=fuse, untargeted=args.untargeted, loss_name=loss_name, adv_init=None)
                label_plus, loss_plus, _ = get_label_loss(adv_np_plus, victim_model, tgt_label, loss_name, targeted = not args.untargeted, device=device)
                n_query += 1
                
                # pretend
                if (not args.untargeted and label_plus != gt_label) and (im_idx not in success_idx_list_pretend):
                    success_idx_list_pretend.add(im_idx)
                    query_list_pretend.append(n_query)

                # stop if successful
                if (not args.untargeted)*(tgt_label == label_plus) or args.untargeted*(tgt_label != label_plus):
                    print('success+', gt_label,' ------> ', label_plus,'\n')
                    success_idx_list.add(im_idx)
                    query_list.append(n_query)
                    loss = loss_plus
                    w_np = w_np_temp_plus
                    adv_np = adv_np_plus
                    w_list.append(w_np.tolist())
                    if label.item() == pred_vic:
                        suc_adv +=1
                    loss_target.append(loss_plus)
                    break

                w_np_temp_minus = w_np.copy()
                w_np_temp_minus[idx_w] -= lr_w * w_inv[idx_w]
                w_np_temp_minus = softmax(w_np_temp_minus)
                adv_np_minus, losses_minus = get_adv_np(im_np, tgt_label, w_np_temp_minus, wb, bound, eps, n_iters, alpha, resize_rate, diversity_prob, algo = algo, fuse=fuse, untargeted=args.untargeted, loss_name=loss_name, adv_init=None)
                label_minus, loss_minus, _ = get_label_loss(adv_np_minus, victim_model, tgt_label, loss_name, targeted = not args.untargeted, device=device)
                n_query += 1

                # pretend
                if (not args.untargeted and label_minus != gt_label) and (im_idx not in success_idx_list_pretend):
                    success_idx_list_pretend.add(im_idx)
                    query_list_pretend.append(n_query)

                # stop if successful
                if (not args.untargeted)*(tgt_label == label_minus) or args.untargeted*(tgt_label != label_minus):
                    print('success-', gt_label,' ------> ', label_minus,'\n')
                    success_idx_list.add(im_idx)
                    query_list.append(n_query)
                    loss = loss_minus
                    w_np = w_np_temp_minus
                    adv_np = adv_np_minus
                    w_list.append(w_np.tolist())
                    if label.item() == pred_vic:
                        suc_adv +=1
                    loss_target.append(loss_minus)
                    break


                if loss_plus < loss_minus:
                    loss = loss_plus
                    w_np = w_np_temp_plus
                    adv_np = adv_np_plus
                    print(n_query,' w',idx_w,' + , loss=', loss.item())
                else:
                    loss = loss_minus
                    w_np = w_np_temp_minus
                    adv_np = adv_np_minus
                    print(n_query,' w',idx_w,' - , loss=', loss.item())
                last_idx = idx_w
                loss_target.append(loss)

                idx_w = (idx_w+1)% n_wb
                
                if n_query > 5 and last_idx == n_wb-1:
                    #lr_w /= 2 # decrease the lr
                    lr_w = lr_w * 0.75
                    print(f"lr_w: {lr_w}")
                
                if n_query >= args.iterw:
                    print('failed ', gt_label,' ------> ', label_idx,' n_query: ',n_query,' \n')
                    query_list.append(n_query)

        rob_acc = 1-(len(success_idx_list)/(im_idx+1))
        suc_rate = 0 if correct_pred==0 else suc_adv / correct_pred
        w_dict = dict(zip(keys, w_np.tolist()))
        metrics = {'robust_acc': rob_acc, 'suc_rate' : suc_rate, 'target_correct_pred': correct_pred, 'n_query': n_query, 'loss':loss_target[-1], 'minloss':min(loss_target), 'maxloss':max(loss_target), 'meanloss':sum(loss_target) / len(loss_target)}
        metrics.update(w_dict)
        experiment.log_metrics(metrics, step=im_idx+1)

    if (args.untargeted):
        print(f"untargeted. total_success: {len(success_idx_list)}; success_rate: {len(success_idx_list)/(im_idx+1)}, avg queries: {np.mean(query_list)}")
    else:
        print(f"targeted. total_success: {len(success_idx_list)}; success_rate: {len(success_idx_list)/(im_idx+1)}, avg queries: {np.mean(query_list)}")

 
    


if __name__ == '__main__':
    main()