from comet_ml import Experiment
import os
import sys
import torch
import copy
import numpy as np
import time, json
import torch.optim as optim

sys.path.append("./MetaAttack_ICLR2020/meta_attack/")

from meta_attack_cifar.attacks.cw_black import BlackBoxLInf
from meta_attack_cifar.data import load_data
from meta_attack_cifar.attacks.generate_gradient import generate_gradient

from meta_attack_cifar.load_attacked_and_meta_model import load_attacked_model, load_meta_model

from meta_attack_cifar.options import args
from app_config import COMET_APIKEY, COMET_WORKSPACE, COMET_PROJECT_BLACK

config = {}
if os.path.exists('config_ids_source_targets.json'):
    with open('config_ids_source_targets.json', 'r') as f:
        config = json.load(f)

experiment = Experiment(
    api_key=COMET_APIKEY,
    project_name=COMET_PROJECT_BLACK,
    workspace=COMET_WORKSPACE,
)

parameters = {'attack': 'Meta', **vars(args), **config}
experiment.log_parameters(parameters)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def main( device):
    use_log = not args.use_zvalue
    is_inception = args.dataset == "imagenet"
    
    # load data and network
    print('Loading %s model and test data' % args.dataset)
    _, test_loader = load_data(args)
    assert test_loader is not None
    
    model = load_attacked_model(args, 0, device)

    meta_model_path = args.load_ckpt
    assert os.path.exists(meta_model_path)
    print('Loading meta model from %s' % meta_model_path)
    meta_model = load_meta_model(args, meta_model_path, device)
    print('Done...')

    if args.attack == 'ours':
        generate_grad = generate_gradient(
                device,
                targeted = not args.untargeted,
                args=args
                )
        attack = BlackBoxLInf(
                targeted = not args.untargeted,
                max_steps = args.steps,
                eps=args.eps,
                search_steps = args.binary_steps,
                cuda = not args.no_cuda,
                )
    
    os.system("mkdir -p {}/{}".format(args.save, args.dataset))

    meta_optimizer = optim.Adam(meta_model.parameters(), lr = args.lr)

    img_no = 0
    total_success = 0
    l2_total = 0.0
    avg_step = 0
    avg_time = 0
    avg_qry = 0

    # acc = save_gradient(model,device,test_loader,mode="test")
    model.eval()
    for i, (img, target) in enumerate(test_loader):
        img, target = img.to(device), target.to(device)
        pred_logit = model(img)
        
        # print('Predicted logits', pred_logit.data[0], '\nProbability', F.softmax(pred_logit, dim = 1).data[0])
        pred_label = pred_logit.argmax(dim=1)
        print('**************************************************\nThe original label before attack', pred_label.item())
        if pred_label != target:
            print("Skip wrongly classified image no. %d, original class %d, classified as %d" % (i, pred_label.item(), target.item()))
            continue
        img_no += 1
        print('img_no',img_no)
        if img_no > args.total_number:
            break
        timestart = time.time()
        
        
        #######################################################################

        meta_model_copy = copy.deepcopy(meta_model)
    
        if not args.untargeted:
            if target ==9 :
                target = torch.tensor([0]).long().cuda()
            else:
                target = torch.tensor([1]).long().cuda() + target
            
 

        #######################################################################

        
        adv, const, first_step,query_bias = attack.run(model, meta_model_copy, img, target, i)
        timeend = time.time()

        if len(adv.shape) == 3:
            adv = adv.reshape((1,) + adv.shape)
        adv = torch.from_numpy(adv).permute(0, 3, 1, 2).cuda()
        #print('adv min and max', torch.min(adv).item(), torch.max(adv).item(), '', torch.min(img).item(), torch.max(img).item())
        diff = (adv-img).cpu().numpy()
        l2_distortion = np.sum(diff**2)**.5
        linf_distortion = np.max(diff)
        print('L2 distortion', l2_distortion)
        print('Linf distortion', linf_distortion)
        
        adv_pred_logit = model(adv)
        adv_pred_label = adv_pred_logit.argmax(dim = 1)
        print('The label after attack', adv_pred_label.item())
        
        success = False
        if args.untargeted:
            if adv_pred_label != target:
                success = True
        else:
            if adv_pred_label == target:
                success = True
        if l2_distortion > 20:
            success = False
        if success:
            total_success += 1
            l2_total += l2_distortion
            avg_step += first_step
            avg_time += timeend - timestart
            #only 1 query for i pixle, because the estimated function is f(x+h)-f(x)/h 
            avg_qry += (first_step-1)//args.finetune_interval*args.update_pixels*1+first_step 
        if total_success == 0:
            pass
        else:
            print("[STATS][L1] total = {}, seq = {}, time = {:.3f}, success = {}, distortion = {:.5f}, avg_step = {:.5f},avg_query = {:.5f}, success_rate = {:.3f}".format(img_no, i, avg_time / total_success, success, l2_total / total_success, avg_step / total_success,avg_qry / total_success, total_success / float(img_no)))
        sys.stdout.flush()
    
        

    if total_success != 0:
        metrics = {'robust_acc': 1-total_success / float(img_no), 'Queries':avg_qry / total_success}
    else:
        metrics = {'robust_acc': 1-total_success / float(img_no), 'Queries':0}
    experiment.log_metrics(metrics, step=1)


if __name__ == "__main__":
    USE_DEVICE = torch.cuda.is_available()
    device = torch.device('cuda' if USE_DEVICE else 'cpu')
    main( device)

