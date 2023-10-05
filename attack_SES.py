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
from utils_attacks import get_adv_np, get_label_loss
from robustbench.utils import load_model
import torch, json, os
import torch.nn as nn
import torchvision.transforms as transforms

from Normalize import Normalize
from app_config import COMET_APIKEY, COMET_WORKSPACE, COMET_PROJECT
import random
from PIL import Image


import os
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
import ImageNet_models as ghost_models


def load_model(model_name, device):
    """Load the model according to the idx in list model_names
    Args: 
        model_name (str): the name of model, chosen from the following list
        model_names = 
    Returns:
        model (torchvision.models): the loaded model
    """
    model = getattr(models, model_name)(pretrained=True)
    model = nn.Sequential(
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        model
    ).to(device).eval()
    return model


def load_model_ghost(model_name, device):
    """Load the model according to the idx in list model_names
    Args: 
        model_name (str): the name of model, chosen from the following list
        model_names = 
    Returns:
        model (torchvision.models): the loaded model
    """
    model = getattr(ghost_models, model_name)(pretrained=False)
    model = nn.Sequential(
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        model
    ).to(device).eval()
    return model


def softmax(vector):
    exp_vector = np.exp(vector)
    normalized_vector = exp_vector / np.sum(exp_vector)
    return normalized_vector

def main():
    parser = argparse.ArgumentParser(description="SES attack ImageNet")

    parser.add_argument("--n_im", type=int, default=10000, help="number of images")
    parser.add_argument('--target', default='vgg19', type=str, help='Target model to use.')
    parser.add_argument("--untargeted", action='store_true', help="run untargeted attack")
    parser.add_argument("--n_models", type=int, default=20, help="number of models")
    parser.add_argument("--dataset", type=str, default=100, help="imagenet1000")

    parser.add_argument("--bound", default='linf', choices=['linf','l2'], help="bound in linf or l2 norm ball")
    parser.add_argument("--eps", type=int, default=16/255, help="perturbation bound: 10 for linf, 3128 for l2")
    parser.add_argument("--iters", type=int, default=10, help="number of inner iterations: 5,6,10,20...")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID: 0,1")
    parser.add_argument("--root", nargs="?", default='result', help="the folder name of result")
    parser.add_argument("--fuse", nargs="?", default='loss', help="the fuse method. loss or logit")
    parser.add_argument("--loss_name", nargs="?", default='cw', help="the name of the loss")
    parser.add_argument("--algo", default='pgd', help="the algo used inside the perturbation machine")
    parser.add_argument("--x", type=int, default=3, help="times alpha by x")
    parser.add_argument("--lr", type=float, default=0.5, help="learning rate of w")
    parser.add_argument("--resize_rate", type=float, default=0.9, help="resize factor used in input diversity")
    parser.add_argument("--diversity_prob", type=float, default=0.5, help="the probability of applying input diversity")
    parser.add_argument("--iterw", type=int, default=50, help="iterations of updating w")
    parser.add_argument('--gamma', default=0.5, type=float)
    parser.add_argument("--imagenet_folder", type=str, default='/home/public/datasets/imagenet/ILSVRC2012/val', help="Path to ImageNet test")
    args = parser.parse_args()


    experiment = Experiment(
        api_key=COMET_APIKEY,
        project_name=COMET_PROJECT,
        workspace=COMET_WORKSPACE,
    )
    experiment.set_name("SES_ImageNet"+"_"+args.target) 

    parameters = {'attack': 'SES', **vars(args), "targeted":False if args.untargeted else True, "dataset":args.dataset}
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


    #load_data

    transform = transforms.Compose([
        transforms.Resize(256),             # Resize the image to 256x256 pixels
        transforms.CenterCrop(224),         # Crop the center 224x224 pixels
        transforms.ToTensor(),              # Convert the image to a PyTorch tensor
        # transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],      # Mean values for normalization
        #     std=[0.229, 0.224, 0.225]        # Standard deviations for normalization
        # )
    ])

    data_dir = args.imagenet_folder

    testset = datasets.ImageFolder(os.path.join(data_dir), transform)
    # print('Validation dataset size:', len(val_datasets))
    testloader = torch.utils.data.DataLoader( torch.utils.data.Subset(testset, range(args.n_im)), batch_size=1, shuffle = False)

     # load surrogate models
    models_names = ['vgg16_bn', 'resnet18', 'squeezenet1_1', 'googlenet', \
                'mnasnet1_0', 'densenet161', 'efficientnet_b0', \
                'regnet_y_400mf', 'resnext101_32x8d', 'convnext_small', \
                'vgg13', 'resnet50', 'densenet201', 'inception_v3', 'shufflenet_v2_x1_0', \
                'mobilenet_v3_small', 'wide_resnet50_2', 'efficientnet_b4', 'regnet_x_400mf', 'vit_b_16']
    models_names = models_names[:args.n_models]
    n_wb = len(models_names)

    wb = []
    for model_name in models_names:
        if 'densenet' in model_name or 'resnet' in model_name:
            print('Using Ghost networks for ',model_name)
            test_ghost = load_model_ghost(model_name, device)
            test = load_model(model_name, device)

            torch.save(test.state_dict(), model_name+'_state_dict.pth')
            test_ghost.load_state_dict(torch.load(model_name+'_state_dict.pth'))

            wb.append(test_ghost)
        else:
            print(f"Load: {model_name}")
            wb.append(load_model(model_name, device))

    
    victim_model = load_model(args.target, device)



    #create folders
    formatted_pairs = [f"{key}-{value}" for key, value in vars(args).items()]
    exp = '_'.join(formatted_pairs)
    adv_root = Path(f"adversarial images/") / exp
    adv_root.mkdir(parents=True, exist_ok=True)
    
    success_idx_list = set()
    success_idx_list_pretend = set() # untargeted success
    correct_pred = 0
    suc_adv = 0
    query_list = []
    query_list_pretend = []
    mean_weights = lambda tab: [sum(col) / len(tab) for col in zip(*tab)] if len(tab) > 0 else []
    w_list = []

    for im_idx,batch in tqdm(enumerate(testloader)):
        loss_target = []
        successful = 0

        if len(batch)==2:
            (image, label) = batch
            tgt_label = None
        else:
            (image, label, tgt_label) = batch
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
        if tgt_label is None:
            tgts_label = [0,20,40,60,80,100]
            if gt_label in tgts_label:
                print('Removing ',gt_label,' from [0,20,40,60,80,100]')
                tgts_label.remove(gt_label)
            tgt_label = random.choice(tgts_label)

        if args.untargeted:
            tgt_label = gt_label
        else:
            print("tgt_label:",tgt_label)

        # start from equal weights
        w_np = np.array([1 for _ in range(len(wb))]) / len(wb)
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
                successful = 1
        else: 
            idx_w = 0         # idx of wb in W
            last_idx = -1    # if no changes after one round, reduce the learning rate
            
            while n_query < args.iterw:
                w_np_temp_plus = w_np.copy()
                w_np_temp_plus[idx_w] += lr_w
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
                        successful = 1
                    loss_target.append(loss_plus)

                    break

                w_np_temp_minus = w_np.copy()
                w_np_temp_minus[idx_w] -= lr_w
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
                        successful = 1
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
        metrics = {'robust_acc': rob_acc, 'suc_rate':successful,'suc_rate_steps' : suc_rate, 'target_correct_pred': correct_pred, 'n_query': n_query, 'loss':loss_target[-1], 'minloss':min(loss_target), 'maxloss':max(loss_target), 'meanloss':sum(loss_target) / len(loss_target)}


        metrics.update(w_dict)
        experiment.log_metrics(metrics, step=im_idx+1)

        #save adv im
        adv_path = adv_root / f"{str(im_idx)}.png"
        image_array = (adv_np * 255).astype(np.uint8)
        image_pil = Image.fromarray(image_array.transpose(1, 2, 0))
        image_pil.save(adv_path)

    if (args.untargeted):
        print(f"untargeted. total_success: {len(success_idx_list)}; success_rate: {len(success_idx_list)/(im_idx+1)}, avg queries: {np.mean(query_list)}")
    else:
        print(f"targeted. total_success: {len(success_idx_list)}; success_rate: {len(success_idx_list)/(im_idx+1)}, avg queries: {np.mean(query_list)}")

 
    


if __name__ == '__main__':
    main()