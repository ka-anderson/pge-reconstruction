import torchvision
from tqdm import tqdm
import torch
from torch.utils.data import random_split
torchvision.disable_beta_transforms_warning()
from typing import Literal
from torch.utils.data import DataLoader

from os.path import join
from os import listdir, mkdir
from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms.v2 as transforms

from datasets.dataset_base import CustomDataset, DOWNLOAD_FOLDER

class DatasetFaceScrub(CustomDataset):
    '''
    Data downloaded from https://www.kaggle.com/datasets/rajnishe/facescrub-full
    The original images vary in size, the default version used here has been scaled to 128x128

    if to_zero_one is true, images are in [0, 1] (default). Otherwise, they are shifted to [-1, 1]
    '''
    def __init__(self, batch_size: int, img_size: int = 128, split: Literal["random", "identity"] = "random", target_to_one_hot=False, num_workers: int = 0, pin_memory=False, to_zero_one=True):
        super().__init__(ds_id='DatasetFaceScrub', batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

        self.img_size = img_size

        if img_size <= 64:
            self.source_img_size = 64
        elif img_size <= 128:
            self.source_img_size = 128
        elif img_size <= 256:
            self.source_img_size = 256
        else:
            self.source_img_size = 0

        self.options.update({
            "class_name": "DatasetFaceScrub", 
            "img_size": img_size,
            "source_img_size": self.source_img_size,
            "target_to_one_hot": target_to_one_hot,
            "to_zero_one": to_zero_one,
            "split": split
        })

        if target_to_one_hot:
            self.dataset = datasets.ImageFolder(
                root=join(DOWNLOAD_FOLDER, "facescrub", str(self.source_img_size)), 
                transform=self._get_transform(to_zero_one),
                # https://stackoverflow.com/questions/63342147/how-to-transform-labels-in-pytorch-to-onehot
                # target_transform=lambda y: torch.zeros(695, dtype=torch.float).scatter_(0, torch.tensor(y), value=1),
                # but torch.Lamdba does not work, and python lambda cannot be distributed
                target_transform=self._to_one_hot,
            )
        else:
            self.dataset = datasets.ImageFolder(
                root=join(DOWNLOAD_FOLDER, "facescrub", str(self.source_img_size)), 
                transform=self._get_transform(to_zero_one),
            )


        if split == "random":
            generator = torch.Generator().manual_seed(0)
            self.dataset_test, self.dataset_train = random_split(self.dataset, [0.05, 0.95], generator=generator)
        elif split == "identity":
            train_indeces = (torch.tensor(self.dataset.targets) <= 470).nonzero(as_tuple=True)[0]
            test_indeces = ((torch.tensor(self.dataset.targets) > 470) & (torch.tensor(self.dataset.targets) <= 510)).nonzero(as_tuple=True)[0]
            valid_indeces = (torch.tensor(self.dataset.targets) > 510).nonzero(as_tuple=True)[0]

            self.dataset_train = torch.utils.data.Subset(self.dataset, train_indeces)
            self.dataset_test = torch.utils.data.Subset(self.dataset, test_indeces)
            self.dataset_valid = torch.utils.data.Subset(self.dataset, valid_indeces)

            self.get_valid_dataloader = lambda: DataLoader(
                    self.dataset_valid,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    drop_last=True,
            )
            

    def output_channels(self):
        return 530

    def _to_one_hot(self, x):
        return torch.zeros(695, dtype=torch.float).scatter_(0, torch.tensor(x), value=1)

    def _get_test_dataset(self) -> datasets.ImageFolder:
        return self.dataset_test
    
    def _get_train_dataset(self) -> datasets.ImageFolder:
        return self.dataset_train

    def _get_transform(self, to_zero_one):
        return_transform = []

        if self.img_size != self.source_img_size:
            return_transform.append(transforms.Resize((self.img_size, self.img_size)))

        return_transform.append(transforms.ToTensor())
        if not to_zero_one:
            return_transform.append(transforms.Normalize(
                    [0.5 for _ in range(3)],
                    [0.5 for _ in range(3)],
                ))

        return transforms.Compose(return_transform)

def save_scaled_version(size):
    mkdir(join(DOWNLOAD_FOLDER, "facescrub", str(size)))
    reshape = transforms.Compose([transforms.Resize((size, size))])
    for path_0 in tqdm(listdir(join(DOWNLOAD_FOLDER, "facescrub", "org"))):
        mkdir(join(DOWNLOAD_FOLDER, "facescrub", str(size), path_0))
        for path_image in listdir(join(DOWNLOAD_FOLDER, "facescrub", "org", path_0)):
            image = Image.open(join(DOWNLOAD_FOLDER, "facescrub", "org", path_0, path_image))
            image = reshape(image)
            image.save(join(DOWNLOAD_FOLDER, "facescrub", str(size), path_0, path_image))

if __name__ == "__main__":
    save_scaled_version(256)