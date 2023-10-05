"""
SES (ours)
"""

import argparse
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets
from Normalize import Normalize

import re
import os
import torch
from torchvision import transforms
from PIL import Image
from robustbench.utils import clean_accuracy


def load_model(model_name, device):
    """Load the model according to the idx in list model_names
    Args: 
        model_name (str): the name of model, chosen from the following list
    Returns:
        model (torchvision.models): the loaded model
    """
    model = getattr(models, model_name)(pretrained=True)
    model = nn.Sequential(
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        model
    ).to(device).eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="calculate robust accuracy")
    parser.add_argument("--target", type=str, default=None, help="the name of the target model")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID: 0,1")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--adversarial_folder", type=str, default='n_im-9_target-vgg19_untargeted-False_bound-linf_eps-0.06274509803921569_iters-10_gpu-0_root-result_fuse-loss_loss_name-cw_algo-pgd_x-3_lr-0.5_resize_rate-0.9_diversity_prob-0.5_iterw-2_gamma-0.5_imagenet_folder-ILSVRC2012_img_val_subset')
    parser.add_argument("--imagenet_folder", type=str, default='/home/public/datasets/imagenet/ILSVRC2012/val', help="Path to ImageNet test to get labels for adversarial images")
    args = parser.parse_args()

    device = f'cuda:{args.gpu}'
    if args.target is None:
        pattern = r'_target-(.*?)_untar'
        model_name = re.findall(pattern, args.adversarial_folder)[0]
        print('Target from Regex: ',model_name)
    else:
        model_name = args.target

    target_model = load_model(model_name, device)


    image_list = []
    # Define a transformation to convert the images to tensors
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image_folder = './adversarial images/'+args.adversarial_folder
    image_files = os.listdir(image_folder)


    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path)
        image_tensor = transform(image)

        image_list.append(image_tensor)

    # Stack the image tensors into a single PyTorch tensor
    x_tensor = torch.stack(image_list)

    print('Adversarial images shape: ',x_tensor.shape)
    x_tensor = x_tensor.to(device)
    num_adv_images = x_tensor.size(0)
    
    #load_data

    transform = transforms.Compose([
        transforms.Resize(256),             # Resize the image to 256x256 pixels
        transforms.CenterCrop(224),         # Crop the center 224x224 pixels
        transforms.ToTensor(),              # Convert the image to a PyTorch tensor
    ])

    data_dir = args.imagenet_folder

    testset = datasets.ImageFolder(os.path.join(data_dir), transform)
    testloader = torch.utils.data.DataLoader( torch.utils.data.Subset(testset, range(num_adv_images)), batch_size=num_adv_images, shuffle = False)
    
    for images,labels in testloader:
        correct_labels = labels.to(device)
    print('Correct labels shape: ',correct_labels.size())


    test_dataset = torch.utils.data.TensorDataset(x_tensor, correct_labels)
    second_loader = torch.utils.data.DataLoader(test_dataset, batch_size= args.batch_size, shuffle=False)

    num_correct_pred = []
    num_im_batch_list = []
    for i, (x_t, y_t) in enumerate(second_loader):
        rob_acc = clean_accuracy(target_model, x_t, y_t)
        print("Robust accuracy of batch ", i,": ",rob_acc)   

        num_correct_pred.append(rob_acc*y_t.size(0))
        num_im_batch_list.append(y_t.size(0))
    print("Total Robust accuracy of",model_name,": ",sum(num_correct_pred)/sum(num_im_batch_list))



if __name__ == '__main__':
    main()