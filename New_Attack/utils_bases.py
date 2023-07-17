import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torchvision.models as models # version 0.12
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


softmax = torch.nn.Softmax(dim=1)


def get_logits_probs(im, model):
    """Get the logits of a model given an image as input
    Args:
        im (PIL.Image or np.ndarray): uint8 image (read from PNG file), for 0-1 float image
        model (torchvision.models): the model returned by function load_model
    Returns:
        logits (numpy.ndarray): direct output of the model
        probs (numpy.ndarray): logits after softmax function
    """
    device = next(model.parameters()).device
    im_tensor = torch.from_numpy(im).unsqueeze(0).to(device) #.float().to(device) 
    logits = model(im_tensor)
    probs = softmax(logits)
    logits = logits.detach().cpu().numpy().squeeze() # convert torch tensor to numpy array
    probs = probs.detach().cpu().numpy().squeeze()
    return logits, probs


def loss_cw(logits, tgt_label, margin=200, targeted=True):
    """c&w loss: targeted
    Args: 
        logits (Tensor): 
        tgt_label (Tensor): 
    """
    device = logits.device
    k = torch.tensor(margin).float().to(device)
    tgt_label = tgt_label
    logits = logits.squeeze()
    onehot_logits = torch.zeros_like(logits)
    onehot_logits[tgt_label] = logits[tgt_label]
    other_logits = logits - onehot_logits
    best_other_logit = torch.max(other_logits)
    tgt_label_logit = logits[tgt_label]
    if targeted:
        loss = torch.max(best_other_logit - tgt_label_logit, -k)
    else:
        loss = torch.max(tgt_label_logit - best_other_logit, -k)
    return loss


class CE_loss(nn.Module):

    def __init__(self, target=False):
        super(CE_loss, self).__init__()
        self.target = target
        self.loss_ce = torch.nn.CrossEntropyLoss()
        
    def forward(self, logits, label):
        loss = self.loss_ce(logits, label)
        if self.target:
            return loss
        else:
            return -loss


class CW_loss(nn.Module):

    def __init__(self, target=False):
        super(CW_loss, self).__init__()
        self.target = target
        self.loss_cw = loss_cw
        
    def forward(self, logits, label):
        return loss_cw(logits, label, targeted=self.target)
        

def get_loss_fn(loss_name, targeted=True):
    """get loss function by name
    Args: 
        loss_name (str): 'cw', 'ce', 'hinge' ...
    """
    if loss_name == 'ce':
        return CE_loss(targeted)
    elif loss_name == 'cw':
        return CW_loss(targeted)


def get_label_loss(im, model, tgt_label, loss_name, targeted=True):
    """Get the loss
    Args:
        im (PIL.Image): uint8 image (read from PNG file)
        tgt_label (int): target label
        loss_name (str): 'cw', 'ce'
    """
    loss_fn = get_loss_fn(loss_name, targeted=targeted)
    logits, _ = get_logits_probs(im, model)
    pred_label = logits.argmax()
    logits = torch.tensor(logits)
    #tgt_label = torch.tensor(tgt_label)
    loss = loss_fn(logits, tgt_label)
    # get top 5 labels
    top5 = logits.argsort()[-5:]
    return pred_label, loss, top5


def get_adv(im, adv, target, w, pert_machine, bound, eps, n_iters, alpha, algo='pgd', fuse='loss', untargeted=False, intermediate=False, loss_name='ce'):
    """Get the adversarial image by attacking the perturbation machine
    Args:
        im (torch.Tensor): original image
        adv (torch.Tensor): initial value of adversarial image, can be a copy of im
        target (torch.Tensor): 0-999
        w (torch.Tensor): weights for models
        pert_machine (list): a list of models
        bound (str): choices=['linf','l2'], bound in linf or l2 norm ball
        eps, n_iters, alpha (float/int): perturbation budget, number of steps, step size
        algo (str): algorithm for generating perturbations
        fuse (str): methods to fuse ensemble methods. logit, prediction, loss
        untargeted (bool): if True, use untargeted attacks, target_idx is the true label.
        intermediate (bool): if True, save the perturbation at every 10 iters.
        loss_name (str): 'cw', 'ce', 'hinge' ...
    Returns:
        adv (torch.Tensor): adversarial output
    """
    # device = next(pert_machine[0].parameters()).device
    n_wb = len(pert_machine)
    if algo == 'mim':
        g = 0 # momentum
        mu = 1 # decay factor
    
    loss_fn = get_loss_fn(loss_name, targeted = not untargeted)
    losses = []
    if intermediate:
        adv_list = []
    for i in range(n_iters):
        adv.requires_grad=True
        input_tensor = adv
        outputs = [model(input_tensor) for model in pert_machine]

        if fuse == 'loss':
            loss = sum([w[idx] * loss_fn(outputs[idx],target) for idx in range(n_wb)])
        elif fuse == 'prob':
            target_onehot = F.one_hot(target, 10)
            prob_weighted = torch.sum(torch.cat([w[idx] * softmax(outputs[idx]) for idx in range(n_wb)], 0), dim=0, keepdim=True)
            loss = - torch.log(torch.sum(target_onehot*prob_weighted))
        elif fuse == 'logit':
            logits_weighted = sum([w[idx] * outputs[idx] for idx in range(n_wb)])
            loss = loss_fn(logits_weighted,target)

        losses.append(loss.item())
        loss.backward()
        
        with torch.no_grad():
            grad = adv.grad
            if algo == 'fgm':
                # needs a huge learning rate
                adv = adv - alpha * grad / torch.norm(grad, p=2)
            elif algo == 'pgd':
                adv = adv - alpha * torch.sign(grad)
            elif algo == 'mim':
                g = mu * g + grad / torch.norm(grad, p=1)
                adv = adv - alpha * torch.sign(g)

            if bound == 'linf':
                # project to linf ball
                adv = (im + (adv - im).clamp(min=-eps,max=eps)).clamp(0,1)
            else:
                # project to l2 ball
                pert = adv - im
                l2_norm = torch.norm(pert, p=2)
                if l2_norm > eps:
                    pert = pert / l2_norm * eps
                adv = (im + pert).clamp(0,1)

        if intermediate and i%10 == 9:
            adv_list.append(adv.detach())
    if intermediate:
        return adv_list, losses
    return adv, losses


def get_adv_np(im_np, target_idx, w_np, pert_machine, bound, eps, n_iters, alpha, algo='mim', fuse='loss', untargeted=False, intermediate=False, loss_name='ce', adv_init=None):
    """Get the numpy adversarial image
    Args:
        im_np (numpy.ndarray): original image
        adv_init (numpy.ndarray): initialization of adv, if None, start with im_np
        target_idx (int): target label
        w_np (list of ints): weights for models
        untargeted (bool): if True, use untargeted attacks, target_idx is the true label.
    Returns:
        adv_np (numpy.ndarray): adversarial output
    """
    device = next(pert_machine[0].parameters()).device
    im = torch.from_numpy(im_np).unsqueeze(0).float().to(device)

    if adv_init is None:
        adv = torch.clone(im) # adversarial image
    else:
        adv = torch.from_numpy(adv_init).unsqueeze(0).float().to(device)
    target = torch.LongTensor([target_idx]).to(device)
    w = torch.from_numpy(w_np).float().to(device)
    adv, losses = get_adv(im, adv, target, w, pert_machine, bound, eps, n_iters, alpha, algo=algo, fuse=fuse, untargeted=untargeted, intermediate=intermediate, loss_name=loss_name)
    if intermediate: # output a list of adversarial images
        adv_np = [adv_.squeeze().cpu().numpy() for adv_ in adv]
    else:
        adv_np = adv.squeeze().cpu().numpy()
    return adv_np, losses
    