import csv
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

def load_imagenet_1000(dataset_root = "imagenet1000"):
    """
    Dataset downloaded form kaggle
    https://www.kaggle.com/datasets/google-brain/nips-2017-adversarial-learning-development-set
    Resized from 299x299 to 224x224

    Args:
        dataset_root (str): root folder of dataset
    Returns:
        img_paths (list of strs): the paths of images
        gt_labels (list of ints): the ground truth label of images
        tgt_labels (list of ints): the target label of images
    """
    dataset_root = Path(dataset_root)
    img_paths = list(sorted(dataset_root.glob('*.png')))
    gt_dict = defaultdict(int)
    tgt_dict = defaultdict(int)
    with open(dataset_root / "images.csv", newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            gt_dict[row['ImageId']] = int(row['TrueLabel'])
            tgt_dict[row['ImageId']] = int(row['TargetClass'])
    gt_labels = [gt_dict[key] - 1 for key in sorted(gt_dict)] # zero indexed
    tgt_labels = [tgt_dict[key] - 1 for key in sorted(tgt_dict)] # zero indexed
    return img_paths, gt_labels, tgt_labels



class ImageNet1000Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataset_root, transform=None):
        """
        Arguments:
            dataset_root (string): Root folder of dataset
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        img_paths, gt_labels, tgt_labels = load_imagenet_1000(dataset_root)
        self.img_paths = img_paths
        self.gt_labels = gt_labels
        self.tgt_labels = tgt_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, im_idx):

        im = Image.open(self.img_paths[im_idx]).convert('RGB')
        image = np.array(im).astype(np.float32) / 255

        if self.transform:
            image = self.transform(image)

        return (image,self.gt_labels[im_idx],self.tgt_labels[im_idx])