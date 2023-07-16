"""
This code is used for NeurIPS 2022 paper "Blackbox Attacks via Surrogate Ensemble Search"

Attack blackbox victim model via querying weight space of ensemble models. 

Blackbox setting
"""

from comet_ml import Experiment
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')

from tqdm import tqdm
from utils_bases import get_adv_np, get_label_loss
from robustbench.utils import load_model
import torch, json, os
import torch.nn as nn
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
#from app_config import COMET_APIKEY, COMET_WORKSPACE, COMET_PROJECT

def main():
    parser = argparse.ArgumentParser(description="BASES attack")

    parser.add_argument('--surrogate_names',  nargs='+', default=['Standard','Ding2020MMA'], help='Surrogate models to use.')
    parser.add_argument('--target_label',  type=int, default=2)
    parser.add_argument("--bound", default='linf', choices=['linf','l2'], help="bound in linf or l2 norm ball")
    parser.add_argument("--eps", type=int, default=8/255, help="perturbation bound: 10 for linf, 3128 for l2")
    parser.add_argument("--iters", type=int, default=10, help="number of inner iterations: 5,6,10,20...")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID: 0,1")
    parser.add_argument("--root", nargs="?", default='result', help="the folder name of result")
    
    parser.add_argument("--fuse", nargs="?", default='loss', help="the fuse method. loss or logit")
    parser.add_argument("--loss_name", nargs="?", default='cw', help="the name of the loss")
    parser.add_argument("--x", type=int, default=3, help="times alpha by x")
    parser.add_argument("--lr", type=float, default=5e-3, help="learning rate of w")
    parser.add_argument("--iterw", type=int, default=50, help="iterations of updating w")
    parser.add_argument("--theta", type=float, default=0.4, help="exploration rate")
    parser.add_argument("--n_im", type=int, default=10, help="number of images")
    parser.add_argument("--untargeted", action='store_true', help="run untargeted attack")
    args = parser.parse_args()

    config = {}
    if os.path.exists('config_ids_source_targets.json'):
        with open('config_ids_source_targets.json', 'r') as f:
            config = json.load(f)

    experiment = Experiment(
        api_key="RDxdOE04VJ1EPRZK7oB9a32Gx",
        project_name="Black-box attack comparison cifar10",
        workspace="meddjilani",
    )

    parameters = {'attack': 'QueryEnsemble', **vars(args), **config}
    experiment.log_parameters(parameters)

    surrogate_names = args.surrogate_names
    n_wb = len(surrogate_names)  

    bound = args.bound
    eps = args.eps
    n_iters = args.iters
    alpha = args.x * eps / n_iters # step-size
    fuse = args.fuse
    loss_name = args.loss_name
    lr_w = float(args.lr)
    device = f'cuda:{args.gpu}'
    

    wb = []
    for model_name in surrogate_names:
        pretrained_model = load_model(model_name, dataset='cifar10', threat_model='Linf')
        pretrained_model.to(device)
        wb.append(pretrained_model)

    # load victim model
    mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    class Normalize(nn.Module):
    
        def __init__(self, mean, std):
            super(Normalize, self).__init__()
            self.mean = mean
            self.std = std
    
        def forward(self, input):
            size = input.size()
            x = input.clone()
            for i in range(size[1]):
                x[:,i] = (x[:,i] - self.mean[i])/self.std[i]
    
            return x
    
    from cifar10_models.resnet import resnet50
    victim_model = resnet50(pretrained=True)
    victim_model = nn.Sequential(
        Normalize(mean, std),
        victim_model
    )
    victim_model.eval()
    victim_model.to(device)


    transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    ])

    testset = CIFAR10(root='../data', train = False, download = True, transform = transform_test)
    testloader = torch.utils.data.DataLoader( torch.utils.data.Subset(testset, range(args.n_im)), batch_size=1, shuffle = False) #sampler=torch.utils.data.sampler.SubsetRandomSampler(range(args.n_im))

    success_idx_list = set()
    success_idx_list_pretend = set() # untargeted success
    query_list = []
    query_list_pretend = []
    mean_weights = lambda tab: [sum(col) / len(tab) for col in zip(*tab)] if len(tab) > 0 else []
    w_list = []
    for im_idx,(image,label) in tqdm(enumerate(testloader)):
        print(f"im_idx: {im_idx + 1}")
        image, label = image.to(device), label.to(device)
        lr_w = float(args.lr) # re-initialize
        image = torch.squeeze(image)
        im_np = np.array(image.cpu())
        print('image mean, 1st value: ',im_np.mean(),', ',np.ravel(im_np)[0])
        gt_label = label
        tgt_label = args.target_label
        if args.untargeted:
            tgt_label = gt_label
        
        # start from equal weights
        if im_idx==0 or len(w_list)==0:
            w_np = np.array([1 for _ in range(len(wb))]) / len(wb)
        else:
            w_np = np.array(mean_weights(w_list)) # average weights of succeded examples
        adv_np, losses = get_adv_np(im_np, tgt_label, w_np, wb, bound, eps, n_iters, alpha, fuse=fuse, untargeted=args.untargeted, loss_name=loss_name, adv_init=None)
        label_idx, loss, _ = get_label_loss(adv_np, victim_model, tgt_label, loss_name, targeted = not args.untargeted)
        n_query = 1
        loss_wb_list = losses   # loss of optimizing wb models
        #print(f"{label_idx}, loss: {loss}") #imagenet_names[label_idx]
        #print(f"w: {w_np.tolist()}")

        # pretend
        if not args.untargeted and label_idx != gt_label:
            success_idx_list_pretend.add(im_idx)
            query_list_pretend.append(n_query)
        if (not args.untargeted and label_idx == tgt_label) or (args.untargeted and label_idx != tgt_label):
            # originally successful
            print('success', tgt_label.item(),' ------> ', label_idx,'\n')
            success_idx_list.add(im_idx)
            query_list.append(n_query)
        else: 
            idx_w = 0         # idx of wb in W
            last_idx = -1    # if no changes after one round, reduce the learning rate
            
            while n_query < args.iterw:
                w_np_temp_plus = w_np.copy()
                w_np_temp_plus[idx_w] += lr_w
                adv_np_plus, losses_plus = get_adv_np(im_np, tgt_label, w_np_temp_plus, wb, bound, eps, n_iters, alpha, fuse=fuse, untargeted=args.untargeted, loss_name=loss_name, adv_init=adv_np)
                label_plus, loss_plus, _ = get_label_loss(adv_np_plus, victim_model, tgt_label, loss_name, targeted = not args.untargeted)
                n_query += 1
                #print(f"query: {n_query}, {idx_w} +, {label_plus}, loss: {loss_plus}") #, imagenet_names[label_plus]
                
                # pretend
                if (not args.untargeted and label_plus != gt_label) and (im_idx not in success_idx_list_pretend):
                    success_idx_list_pretend.add(im_idx)
                    query_list_pretend.append(n_query)

                # stop if successful
                if (not args.untargeted)*(tgt_label == label_plus) or args.untargeted*(tgt_label != label_plus):
                    print('success', tgt_label.item(),' ------> ', label_plus,'\n')
                    success_idx_list.add(im_idx)
                    query_list.append(n_query)
                    loss = loss_plus
                    w_np = w_np_temp_plus
                    adv_np = adv_np_plus
                    loss_wb_list += losses_plus
                    w_list.append(w_np.tolist())
                    break

                w_np_temp_minus = w_np.copy()
                w_np_temp_minus[idx_w] -= lr_w
                adv_np_minus, losses_minus = get_adv_np(im_np, tgt_label, w_np_temp_minus, wb, bound, eps, n_iters, alpha, fuse=fuse, untargeted=args.untargeted, loss_name=loss_name, adv_init=adv_np)
                label_minus, loss_minus, _ = get_label_loss(adv_np_minus, victim_model, tgt_label, loss_name, targeted = not args.untargeted)
                n_query += 1
                #print(f"query: {n_query}, {idx_w} -, {label_minus}, loss: {loss_minus}") #, imagenet_names[label_minus]

                # pretend
                if (not args.untargeted and label_minus != gt_label) and (im_idx not in success_idx_list_pretend):
                    success_idx_list_pretend.add(im_idx)
                    query_list_pretend.append(n_query)

                # stop if successful
                if (not args.untargeted)*(tgt_label == label_minus) or args.untargeted*(tgt_label != label_minus):
                    print('success', tgt_label.item(),' ------> ', label_minus,'\n')
                    success_idx_list.add(im_idx)
                    query_list.append(n_query)
                    loss = loss_minus
                    w_np = w_np_temp_minus
                    adv_np = adv_np_minus
                    loss_wb_list += losses_minus
                    w_list.append(w_np.tolist())
                    break

                # update
                if loss_plus < loss or loss_minus < loss:
                    if loss_plus < loss_minus:
                        loss = loss_plus
                        w_np = w_np_temp_plus
                        adv_np = adv_np_plus
                        loss_wb_list += losses_plus
                        print(n_query,' w',idx_w,' + , loss=', loss.item())
                    else:
                        loss = loss_minus
                        w_np = w_np_temp_minus
                        adv_np = adv_np_minus
                        loss_wb_list += losses_minus
                        print(n_query,' w',idx_w,' - , loss=', loss.item())
                    last_idx = idx_w
                elif last_idx == idx_w: 
                    if np.random.random() < args.theta and lr_w < 0.5:
                        w_np += np.random.normal(0, 1, w_np.shape)
                        lr_w *= 1/0.75 # increase exploration
                    else:
                        lr_w = lr_w * 0.75 # increase exploitation
                    print(f"lr_w: {lr_w} last_idx {last_idx}")

                idx_w = (idx_w+1)% n_wb

                if n_query >= args.iterw:
                    print('failed ', tgt_label.item(),' ------> ', label_idx,' n_query: ',n_query,' \n')
    print('average weights ',w_list)
    if (args.untargeted):
        print(f"untargeted. total_success: {len(success_idx_list)}; success_rate: {len(success_idx_list)/(im_idx+1)}, avg queries: {np.mean(query_list)}")
    else:
        print(f"targeted. total_success: {len(success_idx_list)}; success_rate: {len(success_idx_list)/(im_idx+1)}, avg queries: {np.mean(query_list)}")

    if (not args.untargeted):
        print("When crafting targeted adversarial examples to ",tgt_label)
        print(f"It achived for untargeted attack. total_success: {len(success_idx_list_pretend)}; success_rate: {len(success_idx_list_pretend)/(im_idx+1)}, avg queries: {np.mean(query_list_pretend)}")
    
    print(f"query_list: {query_list}")
    print(f"avg queries: {np.mean(query_list)}")

    rob_acc = 1-(len(success_idx_list)/(im_idx+1))
    metrics = {'robust_acc': rob_acc, 'Queries': np.mean(query_list)}
    experiment.log_metrics(metrics, step=1)




if __name__ == '__main__':
    main()
