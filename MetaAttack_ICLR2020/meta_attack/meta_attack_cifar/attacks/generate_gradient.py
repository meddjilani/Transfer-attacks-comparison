"""PyTorch Carlini and Wagner L2 attack algorithm.

Based on paper by Carlini & Wagner, https://arxiv.org/abs/1608.04644 and a reference implementation at
https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks_tf.py
"""

import numpy as np
import torch.nn.functional as F
from torch import autograd
from .helpers import *


class generate_gradient:

    def __init__(self, device, targeted = False, classes = 10, debug = False, args=None):
        self.debug = debug
        self.targeted = targeted # false
        self.num_classes = classes 
        self.confidence = 0  # FIXME need to find a good value for this, 0 value used in paper not doing much...
        
        self.use_log = True
        self.batch_size = args.update_pixels
        self.device = device     
        self.use_importance = True
        self.constant = 0.5
        
    def _loss(self, output, target, dist, constant):
        real = (target * output).sum(1)
        other = ((1. - target) * output - target * 10000.).max(1)[0]
        if self.targeted:
            if self.use_log:
                loss1 = torch.clamp(torch.log(other + 1e-30) - torch.log(real + 1e-30), min = 0.)
            else:
                loss1 = torch.clamp(other - real + self.confidence, min = 0.)  # equiv to max(..., 0.)
        else:
            if self.use_log:
                loss1 = torch.clamp(torch.log(real + 1e-30) - torch.log(other + 1e-30), min = 0.)
            else:
                loss1 = torch.clamp(real - other + self.confidence, min = 0.)  # equiv to max(..., 0.)
        loss1 = constant * loss1

        # loss2 = dist.squeeze(1)
        # print('loss1 and loss2 is:', loss1, loss2)
        # loss = loss1 + loss2
        # pdb.set_trace()
        loss = loss1
        loss2 = dist
        return loss, loss1, loss2
    
    def run(self, model, img, target, indice):
        
        batch, c, h, w = img.size()
        var_size = c * h * w
        var_list = np.array(range(0, var_size), dtype = np.int32)
        sample_prob = np.ones(var_size, dtype = np.float32) / var_size
  
        ori_img = img
            
        grad = torch.zeros(self.batch_size, dtype = torch.float32)
        modifier = torch.zeros_like(img, dtype = torch.float32)
        
        target_onehot = torch.zeros(target.size() + (self.num_classes,))
        if torch.cuda.is_available():
            target_onehot = target_onehot.to(self.device)
        target_onehot.scatter_(1, target.unsqueeze(1), 1.)
        target_var = autograd.Variable(target_onehot, requires_grad = False) 
        
        img_var = img.repeat(self.batch_size * 2 + 1, 1, 1, 1)
        if self.use_importance:
            var_indice = np.random.choice(var_list.size, self.batch_size, replace = False, p = sample_prob)
        else:
            var_indice = np.random.choice(var_list.size, self.batch_size, replace = False)
        
        # indice = var_list[var_indice]
        #print('indice', indice)
        for i in range(self.batch_size):
            img_var[i*2 + 1].reshape(-1)[indice[i]] += 0.0001
            img_var[i*2 + 2].reshape(-1)[indice[i]] -= 0.0000
        
        output = F.softmax(model(img_var), dim = 1)
        dist = l2_dist(img_var, ori_img, keepdim = True).squeeze(2).squeeze(2)
        # dist = 0
        loss, loss1, loss2 = self._loss(output.data, target_var, dist, self.constant)
        for i in range(self.batch_size):
            grad[i] = (loss[i * 2 + 1] - loss[i * 2 + 2]) / 0.0002
        
        modifier.reshape(-1)[indice] = grad.to(self.device)
        #pdb.set_trace()
        return modifier.cpu().numpy()[0], indice
    
