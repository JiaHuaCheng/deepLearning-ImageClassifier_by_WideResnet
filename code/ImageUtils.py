import torch
import numpy as np
from torchvision import datasets, transforms

"""This script implements the functions for data augmentation
and preprocessing.
"""

def preprocess_image(preprocess_config):
    """
    Args:
        preprocess_config: A dictionary which key, value is args and coressponding setting.
        This dictionary is deinfed under Configure.py

    Returns:
        preprocess_train: transformation for the training images
        preprocess_test: transformation for the testing images

    Note: for mean, std in normalization, I just google the result and use it.
          Reference: https://stackoverflow.com/questions/66678052/how-to-calculate-the-mean-and-the-std-of-cifar10-data
          
          mean: 0.49139968, 0.48215827 ,0.44653124
          std: 0.24703233 0.24348505 0.26158768
    """
    ### YOUR CODE HERE
    # Below, we define the transformation pipeline for both training data and testing data. 
    # We will use random_crop, flip, normalize, cutout.

    c_holes = preprocess_config['cutout_holes']
    c_length = preprocess_config['cutout_length']

    preprocess_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.49139968, 0.48215827 ,0.44653124],
                             std=[0.24703233, 0.24348505, 0.26158768]),
        Cutout(n_holes=c_holes, length=c_length)
    ])

    # we don't know the real mean and std for private test set.
    # Since this value is calculated from a large amount of CIFAR-10 images, we assume it's close to private test set.
    preprocess_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.49139968, 0.48215827 ,0.44653124],
                             std=[0.24703233, 0.24348505, 0.26158768]),
    ])
    
    return preprocess_train, preprocess_test
    ### END CODE HERE

# Other functions
### YOUR CODE HERE
'''
    For implementing Cutout, we refer to this github.  
    (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
'''
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

### END CODE HERE