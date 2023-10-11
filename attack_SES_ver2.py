"""
This code is used for NeurIPS 2022 paper "Blackbox Attacks via Surrogate Ensemble Search"

Attack blackbox victim model via querying weight space of ensemble models. 

Blackbox setting
"""

from comet_ml import Experiment
from app_config import COMET_APIKEY, COMET_WORKSPACE, COMET_PROJECT
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

from utils.class_names_imagenet import lab_dict as imagenet_names
from utils_bases import load_imagenet_1000, load_model, get_adv_np, get_label_loss, load_model_ghost

import torch

def softmax(vector):
    exp_vector = np.exp(vector)
    normalized_vector = exp_vector / np.sum(exp_vector)
    return normalized_vector

def main():
    parser = argparse.ArgumentParser(description="BASES attack")
    parser.add_argument("--victim", nargs="?", default='vgg19', help="victim model")
    parser.add_argument('--ghost', nargs='+', default=['resnet','resnext'])
    parser.add_argument('--random_range', type=float, default=0.22, help='random range of the uniform distribution perturbation in ghost networks')
    parser.add_argument("--n_wb", type=int, default=10, help="number of models in the ensemble: 4,10,20")
    parser.add_argument("--bound", default='linf', choices=['linf','l2'], help="bound in linf or l2 norm ball")
    parser.add_argument("--eps", type=int, default=16, help="perturbation bound: 10 for linf, 3128 for l2")
    parser.add_argument("--iters", type=int, default=10, help="number of inner iterations: 5,6,10,20...")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID: 0,1")
    parser.add_argument("--root", nargs="?", default='result', help="the folder name of result")
    parser.add_argument("--dataset_root", nargs="?", default='imagenet1000/images', help="the folder name of imagenet")
    parser.add_argument("--dataset", type=str, default=100, help="imagenet")
    
    parser.add_argument("--fuse", nargs="?", default='loss', help="the fuse method. loss or logit")
    parser.add_argument("--loss_name", nargs="?", default='cw', help="the name of the loss")
    parser.add_argument("--x", type=int, default=3, help="times alpha by x")
    parser.add_argument("--lr", type=float, default=5e-3, help="learning rate of w")
    parser.add_argument("--iterw", type=int, default=50, help="iterations of updating w")
    parser.add_argument("--n_im", type=int, default=1000, help="number of images")
    parser.add_argument("--untargeted", action='store_true', help="run untargeted attack")
    parser.add_argument("--reduce_lr", action='store_true', help="reduce the learning rate")
    parser.add_argument("--apply_softmax", action='store_true', help="softmax to weights vector")

    args = parser.parse_args()
    

    experiment = Experiment(
        api_key=COMET_APIKEY,
        project_name=COMET_PROJECT,
        workspace=COMET_WORKSPACE,
    )
    experiment.set_name("SES_ver2" + "_" + args.victim)
    parameters = {'attack': 'SES_ver2', **vars(args), "targeted": False if args.untargeted else True,
                  "dataset": args.dataset}
    experiment.log_parameters(parameters)


    n_wb = args.n_wb
    bound = args.bound
    eps = args.eps
    n_iters = args.iters
    alpha = args.x * eps / n_iters # step-size
    fuse = args.fuse
    loss_name = args.loss_name
    lr_w = float(args.lr)
    device = f'cuda:{args.gpu}'
    keys = ['w' + str(i) for i in range(args.n_wb)]

    # load images
    img_paths, gt_labels, tgt_labels = load_imagenet_1000(dataset_path = args.dataset_root)

    # load surrogate models
    surrogate_names = ['vgg16_bn', 'resnet18', 'squeezenet1_1', 'googlenet', \
                'mnasnet1_0', 'densenet161', 'efficientnet_b0', \
                'regnet_y_400mf', 'resnext101_32x8d', 'convnext_small', \
                'vgg13', 'resnet50', 'densenet201', 'inception_v3', 'shufflenet_v2_x1_0', \
                'mobilenet_v3_small', 'wide_resnet50_2', 'efficientnet_b4', 'regnet_x_400mf', 'vit_b_16']
    wb = []
    for model_name in surrogate_names[:n_wb]:
        use_ghost = False
        for ghost_name in args.ghost:
            if ghost_name in model_name.lower():
                use_ghost = True
                break
        if use_ghost:
            print('Using Ghost networks for ', model_name)
            test_ghost = load_model_ghost(model_name, device)
            test = load_model(model_name, device)

            torch.save(test.state_dict(), model_name + '_state_dict.pth')
            test_ghost.load_state_dict(torch.load(model_name + '_state_dict.pth'))

            wb.append(test_ghost)
        else:
            print(f"Load: {model_name}")
            wb.append(load_model(model_name, device))

    # load victim model
    victim_model = load_model(args.victim, device)

    # create exp folders
    exp = f'victim_{args.victim}_{n_wb}wb_{bound}_{eps}_iters{n_iters}_x{args.x}_loss_{loss_name}_lr{lr_w}_iterw{args.iterw}_fuse_{fuse}_v2'
    if args.untargeted:
        exp = 'untargeted_' + exp

    exp_root = Path(f"bb_w_logs/") / exp
    exp_root.mkdir(parents=True, exist_ok=True)
    print(exp)
    adv_root = Path(f"bb_w_adv_images/") / exp
    adv_root.mkdir(parents=True, exist_ok=True)

    # #...
    # ttt = 0
    # for im_idx in range(args.n_im):
    #     im_np = np.array(Image.open(img_paths[im_idx]).convert('RGB'))
    #     gt_label = gt_labels[im_idx]
    #     gt_label_name = imagenet_names[gt_label].split(',')[0]
    #     tgt_label = tgt_labels[im_idx]
        
    #     im = torch.from_numpy(im_np).permute(2,0,1).unsqueeze(0).float().to(device)
    #     print(im.size())
    #     with torch.no_grad():
    #         pred = torch.argmax(victim_model(im)).item()
    #     print('im_idx: ', im_idx, 'pred: ',pred, 'gt_label: ',gt_label)
    #     if pred==gt_label:
    #         ttt =+1
    # print("TOOOTAAAL", (ttt/1000)*100)
    #...
    success_idx_list = set()
    success_idx_list_pretend = set() # untargeted success
    query_list = []
    query_list_pretend = []
    for im_idx in tqdm(range(args.n_im)):
        lr_w = float(args.lr) # re-initialize
        im_np = np.array(Image.open(img_paths[im_idx]).convert('RGB'))
        gt_label = gt_labels[im_idx]
        gt_label_name = imagenet_names[gt_label].split(',')[0]
        tgt_label = tgt_labels[im_idx]
        exp_name = f"idx{im_idx}_f{gt_label}_t{tgt_label}"
        if args.untargeted:
            tgt_label = gt_label
            exp_name = f"idx{im_idx}_f{gt_label}_untargeted"
        
        # start from equal weights
        w_np = np.array([1 for _ in range(len(wb))]) / len(wb)
        adv_np, losses = get_adv_np(im_np, tgt_label, w_np, wb, bound, eps, n_iters, alpha, fuse=fuse, untargeted=args.untargeted, loss_name=loss_name, adv_init=None)
        label_idx, loss, _ = get_label_loss(adv_np/255, victim_model, tgt_label, loss_name, targeted = not args.untargeted)
        n_query = 1
        w_list = []
        loss_wb_list = losses   # loss of optimizing wb models
        loss_bb_list = []       # loss of victim model
        print(f"{label_idx, imagenet_names[label_idx]}, loss: {loss}")

        # pretend
        if not args.untargeted and label_idx != gt_label:
            success_idx_list_pretend.add(im_idx)
            query_list_pretend.append(n_query)

        if (not args.untargeted and label_idx == tgt_label) or (args.untargeted and label_idx != tgt_label):
            # originally successful
            print('success')
            success_idx_list.add(im_idx)
            query_list.append(n_query)
            w_list.append(w_np.tolist())
            loss_bb_list.append(loss)
        else: 
            idx_w = 0         # idx of wb in W
            last_idx = 0    # if no changes after one round, reduce the learning rate
            
            while n_query < args.iterw:
                w_np_temp_plus = w_np.copy()
                w_np_temp_plus[idx_w] += lr_w
                if args.apply_softmax:
                    w_np_temp_plus = softmax(w_np_temp_plus)
                adv_np_plus, losses_plus = get_adv_np(im_np, tgt_label, w_np_temp_plus, wb, bound, eps, n_iters, alpha, fuse=fuse, untargeted=args.untargeted, loss_name=loss_name, adv_init=adv_np)
                label_plus, loss_plus, _ = get_label_loss(adv_np_plus/255, victim_model, tgt_label, loss_name, targeted = not args.untargeted)
                n_query += 1
                print(f"iter: {n_query}, {idx_w} +, {label_plus, imagenet_names[label_plus]}, loss: {loss_plus}")
                
                # pretend
                if (not args.untargeted and label_plus != gt_label) and (im_idx not in success_idx_list_pretend):
                    success_idx_list_pretend.add(im_idx)
                    query_list_pretend.append(n_query)

                # stop if successful
                if (not args.untargeted)*(tgt_label == label_plus) or args.untargeted*(tgt_label != label_plus):
                    print('success')
                    success_idx_list.add(im_idx)
                    query_list.append(n_query)
                    loss = loss_plus
                    w_np = w_np_temp_plus
                    adv_np = adv_np_plus
                    loss_wb_list += losses_plus
                    break

                w_np_temp_minus = w_np.copy()
                w_np_temp_minus[idx_w] -= lr_w
                if args.apply_softmax:
                    w_np_temp_minus = softmax(w_np_temp_minus)
                adv_np_minus, losses_minus = get_adv_np(im_np, tgt_label, w_np_temp_minus, wb, bound, eps, n_iters, alpha, fuse=fuse, untargeted=args.untargeted, loss_name=loss_name, adv_init=adv_np)
                label_minus, loss_minus, _ = get_label_loss(adv_np_minus/255, victim_model, tgt_label, loss_name, targeted = not args.untargeted)
                n_query += 1
                print(f"iter: {n_query}, {idx_w} -, {label_minus, imagenet_names[label_minus]}, loss: {loss_minus}")

                # pretend
                if (not args.untargeted and label_minus != gt_label) and (im_idx not in success_idx_list_pretend):
                    success_idx_list_pretend.add(im_idx)
                    query_list_pretend.append(n_query)

                # stop if successful
                if (not args.untargeted)*(tgt_label == label_minus) or args.untargeted*(tgt_label != label_minus):
                    print('success')
                    success_idx_list.add(im_idx)
                    query_list.append(n_query)
                    loss = loss_minus
                    w_np = w_np_temp_minus
                    adv_np = adv_np_minus
                    loss_wb_list += losses_minus
                    break

                # update
                if loss_plus < loss_minus:
                    loss = loss_plus
                    w_np = w_np_temp_plus
                    adv_np = adv_np_plus
                    loss_wb_list += losses_plus
                    print(f"{idx_w} +")
                    last_idx = idx_w
                else:
                    loss = loss_minus
                    w_np = w_np_temp_minus
                    adv_np = adv_np_minus
                    loss_wb_list += losses_minus
                    print(f"{idx_w} -")
                    last_idx = idx_w
                    
                
                idx_w = (idx_w+1)%n_wb
                if args.reduce_lr:
                    if n_query > 5 and last_idx == n_wb - 1:
                        # lr_w /= 2 # decrease the lr
                        lr_w = lr_w * 0.75
                        print(f"lr_w: {lr_w}")

                w_list.append(w_np.tolist())
                loss_bb_list.append(loss)


        print(f"im_idx: {im_idx}")
        if im_idx in success_idx_list:
            # save to txt
            info = f"im_idx: {im_idx}, iters: {query_list[-1]}, loss: {loss:.2f}, w: {w_np.squeeze().tolist()}\n"
            file = open(exp_root / f'{exp}.txt', 'a')
            file.write(f"{info}")
            file.close()
        print(f"total_success: {len(success_idx_list)}; success_rate: {len(success_idx_list)/(im_idx+1)}, avg queries: {np.mean(query_list)}")

        if im_idx in success_idx_list_pretend:
            # save to txt
            info = f"im_idx: {im_idx}, iters: {query_list_pretend[-1]}, loss: {loss:.2f}, w: {w_np.squeeze().tolist()}\n"
            file = open(exp_root / f'{exp}_pretend.txt', 'a')
            file.write(f"{info}")
            file.close()
        if (not args.untargeted):
            print(f"If it was untargeted. total_success: {len(success_idx_list_pretend)}; success_rate: {len(success_idx_list_pretend)/(im_idx+1)}, avg queries: {np.mean(query_list_pretend)}")


        metrics = {'robust_acc_steps': 1-(len(success_idx_list)/(im_idx+1)),
                   'suc_rate_steps': len(success_idx_list)/(im_idx+1), 'n_query_succesful_steps': np.mean(query_list)}
        w_dict = dict(zip(keys, w_np.tolist()))
        metrics.update(w_dict)
        experiment.log_metrics(metrics, step=im_idx + 1)
        [experiment.log_metric('n_query_succesful',m,  step=im_idx + 1) for m in query_list]


        # save adv image
        adv_path = adv_root / f"{img_paths[im_idx].name}"
        adv_png = Image.fromarray(adv_np.astype(np.uint8))
        adv_png.save(adv_path)

        # plot figs
        fig, ax = plt.subplots(1,5,figsize=(30,5))
        ax[0].plot(loss_wb_list)
        ax[0].set_xlabel('iters')
        ax[0].set_title('loss on surrogate ensemble')
        ax[1].imshow(im_np)
        ax[1].set_title(f"clean image:\n{gt_label_name}")
        victim_label_idx, loss, _ = get_label_loss(adv_np/255, victim_model, tgt_label, loss_name, targeted = not args.untargeted)
        adv_label_name = imagenet_names[victim_label_idx].split(',')[0]
        ax[2].imshow(adv_np/255)
        ax[2].set_title(f"adv image:\n{adv_label_name}")
        ax[3].plot(loss_bb_list)
        ax[3].set_title('loss on victim model')
        ax[3].set_xlabel('iters')
        ax[4].plot(w_list)
        ax[4].legend(surrogate_names[:n_wb], shadow=True, bbox_to_anchor=(1, 1))
        ax[4].set_title('w of surrogate models')
        ax[4].set_xlabel('iters')
        plt.tight_layout()
        if im_idx in success_idx_list:
            plt.savefig(exp_root / f"{exp_name}_success.png")
        else:
            plt.savefig(exp_root / f"{exp_name}.png")
        plt.close()

        # print(f"query_list: {query_list}")
        # print(f"avg queries: {np.mean(query_list)}")
        print("\n\n")


if __name__ == '__main__':
    main()
