"""
SES (ours)
"""

import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets
from Normalize import Normalize



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


def main():
    parser = argparse.ArgumentParser(description="get_correctly_classified")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID: 0,1")
    parser.add_argument("--n_models", type=int, default=21, help="number of surrogate models")
    parser.add_argument("--n_im", type=int, default=5000, help="number of images")
    parser.add_argument("--n_im_correct", type=int, default=100, help="number of images")
    parser.add_argument("--imagenet_folder", type=str, default='/home/public/datasets/imagenet/ILSVRC2012/val', help="Path to ImageNet test")


    args = parser.parse_args()


    device = f'cuda:{args.gpu}'
     # load surrogate models
    models_names = ['densenet121', 'vgg16_bn', 'resnet18', 'squeezenet1_1', 'googlenet', \
                'mnasnet1_0', 'densenet161', 'efficientnet_b0', \
                'regnet_y_400mf', 'resnext101_32x8d', 'convnext_small', \
                'vgg13', 'resnet50', 'densenet201', 'inception_v3', 'shufflenet_v2_x1_0', \
                'mobilenet_v3_small', 'wide_resnet50_2', 'efficientnet_b4', 'regnet_x_400mf', 'vit_b_16']
    models_names = models_names[:args.n_models]
    models_names = models_names + ['vgg19', 'densenet121', 'resnext50_32x4d']
    
    wb = []
    for model_name in models_names:
        print(f"load: {model_name}")
        wb.append(load_model(model_name, device))


    #load data
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

    val_datasets = datasets.ImageFolder(os.path.join(data_dir), transform)
    # print('Validation dataset size:', len(val_datasets))
    val_loader = torch.utils.data.DataLoader( torch.utils.data.Subset(val_datasets, range(args.n_im)), batch_size=1, shuffle = False)


    correct_classified_idx = []
    for im_idx,(image,label) in enumerate(val_loader):
        if len(correct_classified_idx) >= args.n_im_correct:
            print(args.n_im_correct,' correctly classified images have been found. Finished ! ')
            break
        print(f"\nim_idx: {im_idx + 1}")
        image, label = image.to(device), label.to(device)
        correct_classified_all_models = True
        with torch.no_grad():
            for model in wb:
                pred_vic = torch.argmax(model(image)).item()
                print('Label:',label.item(),', Prediction:' ,pred_vic)
                if label.item() != pred_vic:
                    correct_classified_all_models = False
                    break
        if correct_classified_all_models:
            print(f"im_idx: {im_idx + 1} ADDED")           
            correct_classified_idx.append(im_idx)
        
    
    correct_classified_idx_as_string = str(correct_classified_idx)

    file_path = "correct_classified_idx.txt"
    with open(file_path, 'w') as file:
        file.write(correct_classified_idx_as_string)



if __name__ == '__main__':
    main()