from __future__ import print_function
import argparse
import os
import math
import sys
import random
import numpy as np
import foolbox as fb
from foolbox.criteria import Misclassification, TargetedMisclassification

import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.nn.functional import mse_loss
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.utils.data.sampler as sp
from vgg import VGG #medben
from resnet import resnet50
import robustbench #medben 
import json


if __name__ == '__main__':
    cudnn.benchmark = True

    nz = 128
    target =False
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0" #medben

    SEED = 1000
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    class Logger(object):
        def __init__(self, filename='default.log', stream=sys.stdout):
            self.terminal = stream
            self.log = open(filename, 'a')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            pass

    sys.stdout = Logger('dast_cifar10.log', sys.stdout)

    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=500, help='input batch size')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--niter', type=int, default=20, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', default=True, action='store_true', help='enables cuda')
    parser.add_argument('--alpha', type=float, default=0.2, help='alpha')
    parser.add_argument('--beta', type=float, default=0.1, help='beta')#(from 0.1 to 20.0)--DasTP     0.0--DasTL
    parser.add_argument('--G_type', type=int, default=1, help='G type')
    parser.add_argument('--save_folder', type=str, default='saved_model', help='alpha')
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu") #medben
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    transforms = transforms.Compose([transforms.ToTensor()])

    testset = torchvision.datasets.CIFAR10(root='./cifar10 data', train=False,
                                        download=True,
                                        transform=transforms
                                        )
    netD = resnet50()
    netD.to(device)
    netD.eval()
    print(device)

    original_net = robustbench.utils.load_model(model_name='Standard', dataset='cifar10', threat_model='Linf')
    original_net.to(device)
    print('originalnet parm',next(original_net.parameters()).is_cuda )

    original_net.eval()

    fmodel = fb.PyTorchModel(netD, bounds=(0.0,1.0))
    attack_fb = fb.attacks.L2BasicIterativeAttack(abs_stepsize=0.01, steps=240, random_start=False)

    nc=3

    #data_list = [i for i in range(6000, 7000)] # fast validation #medben 6k,8k
    with open('config_ids_source_targets.json','r') as f:
        config = json.load(f)
    ids = config['ids']
    x,y = robustbench.data.load_cifar10()
    x,y = x[ids,:,:,:], y[ids]

    class ImageDataset(torch.utils.data.Dataset):
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels

        def __len__(self):
            return len(self.images)

        def __getitem__(self, index):
            return self.images[index], self.labels[index]

    testset = ImageDataset(x, y)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, num_workers=2)


    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    class Loss_max(nn.Module):
        def __init__(self):
            super(Loss_max, self).__init__()
            return

        def forward(self, pred, truth, proba):
            criterion_1 = nn.MSELoss()
            criterion = nn.CrossEntropyLoss()
            pred_prob = F.softmax(pred, dim=1)
            loss = criterion(pred, truth) + criterion_1(pred_prob, proba) * opt.beta
            # loss = criterion(pred, truth)
            final_loss = torch.exp(loss * -1)
            return final_loss

    def get_att_results(model, target):
        correct = 0.0
        total = 0.0
        total_L2_distance = 0.0
        att_num = 0.
        acc_num = 0.
        for data in testloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            if target:
                # randomly choose the specific label of targeted attack
                labels = torch.randint(0, 9, (inputs.size(0),)).to(device)
                # test the images which are not classified as the specific label

                ones = torch.ones_like(predicted)
                zeros = torch.zeros_like(predicted)
                acc_sign = torch.where(predicted == labels, zeros, ones)
                acc_num += acc_sign.sum().float()
                # adv_inputs_ori = adversary.perturb(inputs, labels)
                _, adv_inputs_ori, _ = attack_fb(fmodel, inputs, TargetedMisclassification(labels), epsilons=1.5)
                L2_distance = (adv_inputs_ori - inputs).squeeze()
                # L2_distance = (torch.linalg.norm(L2_distance, dim=list(range(1, inputs.squeeze().dim())))).data
                L2_distance = (torch.linalg.norm(L2_distance.flatten(start_dim=1), dim=1)).data
                # L2_distance = (torch.linalg.matrix_norm(L2_distance, dim=0, keepdim=True)).data
                L2_distance = L2_distance * acc_sign
                total_L2_distance += L2_distance.sum()
                with torch.no_grad():
                    outputs = model(adv_inputs_ori)
                    _, predicted = torch.max(outputs.data, 1)

                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                    att_sign = torch.where(predicted == labels, ones, zeros)
                    att_sign = att_sign + acc_sign
                    att_sign = torch.where(att_sign == 2, ones, zeros)
                    att_num += att_sign.sum().float()
            else:
                ones = torch.ones_like(predicted)
                zeros = torch.zeros_like(predicted)
                acc_sign = torch.where(predicted == labels, ones, zeros)
                acc_num += acc_sign.sum().float()
                print('inputs and labels and original_net',inputs.is_cuda, labels.is_cuda, next(original_net.parameters()).is_cuda )
                _, adv_inputs_ori, _ = attack_fb(fmodel, inputs, Misclassification(labels), epsilons=1.5)
                L2_distance = (adv_inputs_ori - inputs).squeeze()
                L2_distance = (torch.linalg.norm(L2_distance.flatten(start_dim=1), dim=1)).data
                # L2_distance = (torch.linalg.matrix_norm(L2_distance, dim=0, keepdim=True)).data
                L2_distance = L2_distance * acc_sign
                total_L2_distance += L2_distance.sum()
                with torch.no_grad():
                    outputs = model(adv_inputs_ori)
                    _, predicted = torch.max(outputs.data, 1)

                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                    att_sign = torch.where(predicted == labels, zeros, ones)
                    att_sign = att_sign + acc_sign
                    att_sign = torch.where(att_sign == 2, ones, zeros)
                    att_num += att_sign.sum().float()

        if target:
            att_result = (att_num / acc_num * 100.0)
            print('Attack success rate: %.2f %%' %
                  ((att_num / acc_num * 100.0)))
        else:
            att_result = (att_num / acc_num * 100.0)
            print('Attack success rate: %.2f %%' %
                  (att_num / acc_num * 100.0))
        print('l2 distance:  %.4f ' % (total_L2_distance / acc_num))
        return att_result


    class pre_conv(nn.Module):
        def __init__(self, num_class):
            super(pre_conv, self).__init__()
            self.nf = 64
            if opt.G_type == 1:
                self.pre_conv = nn.Sequential(
                    nn.Conv2d(nz, self.nf * 2, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(self.nf * 2),
                    # nn.LeakyReLU(0.2, inplace=True),
                    nn.ReLU(True),

                    nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(self.nf * 2),
                    # nn.LeakyReLU(0.2, inplace=True),
                    nn.ReLU(True),

                    nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(self.nf * 2),
                    # nn.LeakyReLU(0.2, inplace=True),
                    nn.ReLU(True),

                    nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(self.nf * 2),
                    # nn.LeakyReLU(0.2, inplace=True),
                    nn.ReLU(True),

                    nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(self.nf * 2),
                    # nn.LeakyReLU(0.2, inplace=True),
                    nn.ReLU(True),

                    nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(self.nf * 2),
                    # nn.LeakyReLU(0.2, inplace=True)
                    nn.ReLU(True),
                )
            elif opt.G_type == 2:
                self.pre_conv = nn.Sequential(
                    nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, round((self.shape[0]-1) / 2), bias=False),
                    nn.BatchNorm2d(self.nf * 8),
                    nn.ReLU(True),  # added

                    # nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, 1, bias=False),
                    # nn.BatchNorm2d(self.nf * 8),
                    # nn.ReLU(True),

                    nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, round((self.shape[0]-1) / 2), bias=False),
                    nn.BatchNorm2d(self.nf * 8),
                    nn.ReLU(True),

                    nn.Conv2d(self.nf * 8, self.nf * 4, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(self.nf * 4),
                    nn.ReLU(True),

                    nn.Conv2d(self.nf * 4, self.nf * 2, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(self.nf * 2),
                    nn.ReLU(True),

                    nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(self.nf),
                    nn.ReLU(True),

                    nn.Conv2d(self.nf, self.shape[0], 3, 1, 1, bias=False),
                    nn.BatchNorm2d(self.shape[0]),
                    nn.ReLU(True),

                    nn.Conv2d(self.shape[0], self.shape[0], 3, 1, 1, bias=False),
                    # if self.shape[0] == 3:
                    #     nn.Tanh()
                    # else:
                    nn.Sigmoid()
                )
        def forward(self, input):
            output = self.pre_conv(input)
            return output

    pre_conv_block = []
    for i in range (10):
        pre_conv_block.append(pre_conv(10).to(device))

    class Generator(nn.Module):
        def __init__(self, num_class):
            super(Generator, self).__init__()
            self.nf = 64
            self.num_class = num_class
            if opt.G_type == 1:
                self.main = nn.Sequential(
                    nn.Conv2d(self.nf * 2, self.nf * 4, 3, 1, 0, bias=False),
                    nn.BatchNorm2d(self.nf * 4),
                    nn.LeakyReLU(0.2, inplace=True),

                    # nn.Conv2d(self.nf * 4, self.nf * 4, 3, 1, 1, bias=False),
                    # nn.BatchNorm2d(self.nf * 4),
                    # nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(self.nf * 4, self.nf * 8, 3, 1, 0, bias=False),
                    nn.BatchNorm2d(self.nf * 8),
                    nn.LeakyReLU(0.2, inplace=True),

                    # nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, 1, bias=False),
                    # nn.BatchNorm2d(self.nf * 8),
                    # nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(self.nf * 8, self.nf * 4, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(self.nf * 4),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(self.nf * 4, self.nf * 2, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(self.nf * 2),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(self.nf),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(self.nf, nc, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(nc),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(nc, nc, 3, 1, 1, bias=False),
                    nn.Sigmoid()
                )
            elif opt.G_type == 2:
                self.main = nn.Sequential(
                    nn.Conv2d(nz, self.nf * 2, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(self.nf * 2),
                    nn.ReLU(True),

                    nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(self.nf * 2),
                    nn.ReLU(True),

                    nn.ConvTranspose2d(self.nf * 2, self.nf * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(self.nf * 4),
                    nn.ReLU(True),

                    nn.ConvTranspose2d(self.nf * 4, self.nf * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(self.nf * 4),
                    nn.ReLU(True),

                    nn.ConvTranspose2d(self.nf * 4, self.nf * 8, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(self.nf * 8),
                    nn.ReLU(True),

                    nn.ConvTranspose2d(self.nf * 8, self.nf * 8, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(self.nf * 8),
                    nn.ReLU(True),

                    nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(self.nf * 8),
                    nn.ReLU(True)
                )
        def forward(self, input):
            output = self.main(input)
            return output


    class Generator_cifar10(nn.Module):
        def __init__(self, num_class):
            super(Generator_cifar10, self).__init__()
            self.nf = 64
            self.num_class = num_class
            if opt.G_type == 1:
                self.main = nn.Sequential(                   #128, 32, 32
                    nn.Conv2d(128, 256, 3, 1, 1, bias=False),          #64 32 32
                    nn.BatchNorm2d(256),
                    # nn.LeakyReLU(0.2, inplace=True),
                    nn.ReLU(True),

                    # nn.Conv2d(self.nf * 4, self.nf * 4, 3, 1, 1, bias=False),
                    # nn.BatchNorm2d(self.nf * 4),
                    # nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(256, 512, 3, 1, 1, bias=False),        #32 32 32
                    nn.BatchNorm2d(512),
                    # nn.LeakyReLU(0.2, inplace=True),
                    nn.ReLU(True),

                    # nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, 1, bias=False),
                    # nn.BatchNorm2d(self.nf * 8),
                    # nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(512, 256, 3, 1, 1, bias=False),          #16 32 32
                    nn.BatchNorm2d(256),
                    # nn.LeakyReLU(0.2, inplace=True),
                    nn.ReLU(True),

                    nn.Conv2d(256, 128, 3, 1, 1, bias=False),         #8 32 32
                    nn.BatchNorm2d(128),
                    # nn.LeakyReLU(0.2, inplace=True),
                    nn.ReLU(True),

                    nn.Conv2d(128, 64, 3, 1, 1, bias=False),         #4 32 32
                    nn.BatchNorm2d(64),
                    # nn.LeakyReLU(0.2, inplace=True),
                    nn.ReLU(True),

                    nn.Conv2d(64, 3, 3, 1, 1, bias=False),     #2 32 32
                    nn.BatchNorm2d(3),
                    # nn.LeakyReLU(0.2, inplace=True),
                    nn.ReLU(True),

                    nn.Conv2d(3, 3, 3, 1, 1, bias=False),   #1 28 28--->3 32 32
                # nn.BatchNorm2d(3),#---------
                    nn.Sigmoid()
                )
            elif opt.G_type == 2:
                self.main = nn.Sequential(
                    nn.Conv2d(nz, self.nf * 2, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(self.nf * 2),
                    nn.ReLU(True),

                    nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(self.nf * 2),
                    nn.ReLU(True),

                    nn.ConvTranspose2d(self.nf * 2, self.nf * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(self.nf * 4),
                    nn.ReLU(True),

                    nn.ConvTranspose2d(self.nf * 4, self.nf * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(self.nf * 4),
                    nn.ReLU(True),

                    nn.ConvTranspose2d(self.nf * 4, self.nf * 8, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(self.nf * 8),
                    nn.ReLU(True),

                    nn.ConvTranspose2d(self.nf * 8, self.nf * 8, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(self.nf * 8),
                    nn.ReLU(True),

                    nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(self.nf * 8),
                    nn.ReLU(True)
                )
        def forward(self, input):
            output = self.main(input)
            return output

    def chunks(arr, m):
        n = int(math.ceil(arr.size(0) / float(m)))
        return [arr[i:i + n] for i in range(0, arr.size(0), n)]

    netG = Generator_cifar10(10)
    netG.to(device)

    criterion = nn.CrossEntropyLoss()
    criterion_max = Loss_max()

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr*2.0, betas=(opt.beta1, 0.999))

    optimizer_block = []
    for i in range(10):
        optimizer_block.append(optim.Adam(pre_conv_block[i].parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)))

    with torch.no_grad():
        correct_netD = 0.0
        total = 0.0
        netD.eval()
        for data in testloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = original_net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_netD += (predicted == labels).sum()
        print('original net accuracy: %.2f %%' %
                (100. * correct_netD.float() / total))

    att_result = get_att_results(original_net, target=False)
    print('Attack success rate: %.2f %%' %
            (att_result))

    batch_num = 100
    best_accuracy = 0.0
    best_att = 0.0
    cnt =0

    for epoch in range(opt.niter):
        print('-------------------train D-----------------')
        netD.train()

        for ii in range(batch_num):
            netD.zero_grad()

            ############################
            # (1) Update D network:
            ###########################
            noise = torch.rand(opt.batchSize, nz, 1, 1, device=device).to(device)
            noise_chunk = chunks(noise, 10)
            for i in range(len(noise_chunk)):
                tmp_data = pre_conv_block[i](noise_chunk[i])
                gene_data = netG(tmp_data)
                label = torch.full((noise_chunk[i].size(0),), i).to(device)
                if i == 0:
                    data = gene_data
                    set_label = label
                else:
                    data = torch.cat((data, gene_data), 0)
                    set_label = torch.cat((set_label, label), 0)

            index = torch.randperm(set_label.size()[0])
            data = data[index]
            set_label = set_label[index]

            # obtain the output label of T
            with torch.no_grad():
                # outputs = original_net(data)
                outputs = original_net(data)
                cnt += 1
                _, label = torch.max(outputs.data, 1)
                outputs = F.softmax(outputs, dim=1)
            output = netD(data.detach())
            prob = F.softmax(output, dim=1)
            # print(torch.sum(outputs) / 500.)
            errD_prob = mse_loss(prob, outputs, reduction='mean')
            errD_fake = criterion(output, label) + errD_prob * opt.beta
            # D_G_z1 = errD_fake.mean().item()
            errD_fake.backward()

            errD = errD_fake
            # if errD.item() > 0.3:
            optimizerD.step()

            del output, errD_fake

            ############################
            # (2) Update G network:
            ###########################
            netG.zero_grad()
            for i in range(10):
                pre_conv_block[i].zero_grad()
            output = netD(data)
            loss_imitate = criterion_max(pred=output, truth=label, proba=outputs)
            loss_diversity = criterion(output, label.squeeze().long())
            # alpha = random.uniform(0, 0.2)
            # errG = opt.alpha * loss_diversity + loss_imitate
            errG = opt.alpha * loss_diversity + loss_imitate
            # errG = loss_diversity
            if loss_diversity.item() <= 0.2:
                opt.alpha = loss_diversity.item()
            errG.backward()
            # optimizerG.step()
            # for i in range(10):
            #     optimizer_block[i].step()

            if (ii % 20) == 0:
                print('[%d/%d][%d/%d] D: %.4f D_prob: %.4f loss_imitate: %.4f loss_diversity: %.4f'
                    % (epoch, opt.niter, ii, batch_num,
                        errD.item(), errD_prob.item(), loss_imitate.item(), loss_diversity.item()))
                print('current opt.alpha: ', opt.alpha)

        netD.eval()
        att_result = get_att_results(original_net, target=False)
        # print('Attack success rate: %.2f %%' % (att_result))

        if best_att < att_result:
            torch.save(netD.state_dict(),
                            opt.save_folder+'/resnet50_netD_epoch_%d.pth' % (epoch))
            torch.save(netG.state_dict(),
                            opt.save_folder+'/resnet50_netG_epoch_%d.pth' % (epoch))
            best_att = att_result
            print('model saved')
        else:
            print('Best ASR: %.2f %%' % (best_att))

        ################################################
        # evaluate the accuracy of trained D:
        ################################################
        with torch.no_grad():
            correct_netD = 0.0
            total = 0.0
            netD.eval()
            for data in testloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = netD(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct_netD += (predicted == labels).sum()
            print('substitute accuracy: %.2f %%' %
                    (100. * correct_netD.float() / total))
            if best_accuracy < correct_netD:
                torch.save(netD.state_dict(),
                        opt.save_folder+'/resnet50_netD_epoch_%d.pth' % (epoch))
                torch.save(netG.state_dict(),
                        opt.save_folder+'/resnet50_netG_epoch_%d.pth' % (epoch))
                best_accuracy = correct_netD
                print('model saved')
            else:
                print('Best ACC: %.2f %%' % (100. * best_accuracy.float() / total))