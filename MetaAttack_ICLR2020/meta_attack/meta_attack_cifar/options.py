
import argparse
import random 
import numpy as np
import torch
#python scripts
__author__='Du Jiawei NUS/IHPC'
__email__='dujw@ihpc.a-star.edu'
#Descrption:
def str2bool(s):
    assert s in ['True', 'False']
    if s == 'True':
        return True
    else:
        return False


parser = argparse.ArgumentParser()


#important parameters 
parser.add_argument("--n_examples", type = int, default = 3)

parser.add_argument("--steps", type = int, default = 1000, help = "set 0 to use default value")
parser.add_argument('--eps', type=float, default=8 / 255)
parser.add_argument("--max_fintune_iter", type = int, default = 63, help = "maximum finetune iterations")

parser.add_argument("--finetune_interval", type = int, default = 3, help = "iteration interval for finetuneing")

parser.add_argument("--model", type = str, default = 'Carmon2019Unlabeled', help = 'the model selected to be used for meta-model')
parser.add_argument("--target", type = str, default = 'Carmon2019Unlabeled', help = 'the model selected to be attaked')
parser.add_argument('--learning_rate', default = 1e-2, type = float, help = 'learning rate')
parser.add_argument('--update_pixels', default = 125, type = int, help = 'updated pixels every iteration')
parser.add_argument('--simba_update_pixels', default = 125, type = int, help = 'updated pixels every iteration')
parser.add_argument('--total_number', default = 1000, type = int, help = 'maxximum attack numbers')
parser.add_argument("--untargeted", type = str, default = 'True')
parser.add_argument("--istransfer", type = str, default = 'False')

parser.add_argument("--load_ckpt", default = "./MetaAttack_ICLR2020/checkpoints/meta_attacker/cifar/0.7234403cifar_VGG.pt", help = "path to meta attacker model")



# the following is default setting, no need to modify to normal testing

parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--test_batch_size', type = int, default = 1)
parser.add_argument('--no_cuda', action = 'store_true')

parser.add_argument("--attack", choices = ["ours"], default = "ours")
parser.add_argument("--dataset", choices=["mnist", "cifar10", "imagenet"], default = "cifar10")

parser.add_argument("-s", "--save", default = "./saved_results")
parser.add_argument("-n", "--numimg", type = int, default = 0, help = "number of test images to attack")
parser.add_argument("-p", "--print_every", type = int, default = 100, help = "print objs every PRINT_EVERY iterations")
parser.add_argument("-f", "--firstimg", type = int, default = 0)
parser.add_argument("-b", "--binary_steps", type = int, default = 0)
parser.add_argument("-c", "--init_const", type = float, default = 0.0)
parser.add_argument("-z", "--use_zvalue", action = 'store_true')
parser.add_argument("-r", "--reset_adam", action = 'store_true', help = "reset adam after an initial solution is found")
parser.add_argument("--use_resize", action = 'store_true', help = "resize image (only works on imagenet!)")
parser.add_argument("--seed", type = int, default = 1216)
parser.add_argument("--solver", choices = ["adam", "newton", "adam_newton", "fake_zero"], default = "adam")
parser.add_argument("--save_ckpts", default = "", help = "path to save checkpoint file")
parser.add_argument("--start_iter", default = 0, type = int, help = "iteration number for start, useful when loading a checkpoint")
parser.add_argument("--init_size", default = 32, type = int, help = "starting with this size when --use_resize")
parser.add_argument("--uniform", action = 'store_true', help = "disable importance sampling")

parser.add_argument('--lr', default = 1e-2, type = int, help = 'learning rate')
parser.add_argument('--inception', action = 'store_true', default = False)
parser.add_argument('--use_tanh', action = 'store_true', default = True)

parser.add_argument('--debug', action = 'store_true', default = True)

args = parser.parse_args()
# True False process
vars(args)['istransfer'] = str2bool(args.istransfer)
vars(args)['untargeted'] = str2bool(args.untargeted)
#max iteration process
if args.untargeted:
    args.maxiter = 100
else:
    args.maxiter = 300

if args.binary_steps != 0:
    args.init_const = 0.01
else:
    args.init_const = 0.5

if args.binary_steps == 0:
    args.binary_steps = 1

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)



