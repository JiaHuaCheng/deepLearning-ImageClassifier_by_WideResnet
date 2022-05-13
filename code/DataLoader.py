import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from ImageUtils import preprocess_image
from PIL import Image

""" This script implements the functions for reading data.

    To load our private_test_images.npy, we refer to this post.
    (https://discuss.pytorch.org/t/how-to-get-npy-dataloader/52379)

"""

class Private_Test_Dataset(Dataset):
    
    # np_file_paths should be ./data
    def __init__(self, root, transform=None):
        private_data = 'private_test_images_v3.npy'
        private_test_path = os.path.join(root, private_data)
        # input shape is 2000, 3072, we need to reshape to 2000, 32, 32, 3
        self.data = np.load(private_test_path).reshape(-1, 32, 32, 3)
        self.transform = transform # this should be 1. toTorsor, 2. normalize

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.data[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.data)


def load_data(data_dir, preprocess_configs):
    """Load the CIFAR-10 dataset.
    Args:
        data_dir: the directory where CIFAR-10 saved
        preprocess_config: A dictionary which include all arguements and settings for preprocessing data.

    Returns:
        training_loder: return iterable object which length is 50000 / batch_size
        validation_loader: return iterable object which length is 10000 / batch_size
    """

    ### YOUR CODE HERE
    # the CIFAR-10 data is aviable under torchvision.datasets.CIFAR10
    # We need to define transform for processing CIFAR10 raw data, and this transform pipeline is defined under ImageUtils.py
    transform_train, transform_public_test = preprocess_image(preprocess_configs)

    # we create two CIFAR-10 dataset for training, validation. 
    # recall from HW1, we can use both training and validation to train model. The key point is not to peek test data.
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=transform_train,
        download=True
    )

    public_test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        transform=transform_public_test, # we don't need to flip and crop for testing.
        download=True
    )

    training_loader = DataLoader(
        dataset=train_dataset,
        batch_size=preprocess_configs['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    public_testing_loader = DataLoader(
        dataset=public_test_dataset,
        batch_size=preprocess_configs['batch_size'],
        shuffle=False, # no randomness during testing.
        num_workers=0,
        pin_memory=True,
    )
    # print(len(training_loader), len(public_testing_loader)) # return batch data object. 
    return training_loader, public_testing_loader
    ### END CODE HERE


def load_testing_images(data_dir, preprocess_config):
    """Load the CIFAR-10 dataset.
    Args:
        data_dir: the directory where private test data saved
        preprocess_config: A dictionary which include all arguements and settings for preprocessing data.

    Returns:
        training_loder: return iterable object which length is 50000 / batch_size
        validation_loader: return iterable object which length is 10000 / batch_size
    """

    ### YOUR CODE HERE
    _, transform_private_test = preprocess_image(preprocess_config)

    # use cut-off to augment data.
    private_test_dataset = Private_Test_Dataset(
        root=data_dir,
        transform=transform_private_test
    )

    private_test_loader = DataLoader(
        dataset=private_test_dataset,
        batch_size=preprocess_config['batch_size'],
        shuffle=False, # no randomness during testing.
        num_workers=0,
        pin_memory=True
    )

    return private_test_loader
    ### END CODE HERE