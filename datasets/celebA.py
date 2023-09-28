from os import mkdir
import torch
import torchvision
from tqdm import tqdm
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

from misc_helpers.helpers import repo_dir
torchvision.disable_beta_transforms_warning()

import torchvision.datasets as datasets
import torchvision.transforms.v2 as transforms
from os.path import join, isfile, isdir
from os import listdir, mkdir, rename, getcwd
from PIL import Image


from datasets.dataset_base import CustomDataset, DOWNLOAD_FOLDER

# zero mean and unit sd
# https://stackoverflow.com/questions/63746182/correct-way-of-normalizing-and-scaling-the-mnist-dataset
MNIST_NORM = transforms.Normalize((0.1307,), (0.3081,))
IMAGE_NORM = transforms.Normalize([0.5], [0.5])

OUTPUT_CHANNELS = {
    "attr": 40,
    "identity": 10177,
    "bbox": 4,
    "landmarks": 10,
    "landm_bb": 14,
}

class DatasetCelebA(CustomDataset):
    '''
    * an image size other than 256 results in the images being rescaled using transforms.Resize. This is done AFTER the crop.
    * If crop and image size are left at the default (178, 256), it uses a prepared dataset and no further transformations are needed, which is much faster.
    * default target type: attr, (Tensor shape=(40,) dtype=int): binary (0, 1) labels for attributes (see https://pytorch.org/vision/main/generated/torchvision.datasets.CelebA.html for other target types)
    * if to_zero_one is true, images are in [0, 1] (default). Otherwise, they are shifted to [-1, 1]

    '''
    def __init__(self, batch_size: int, center_crop=178, target_type: Literal["attr", "identity", "bbox", "landmarks", "landm_bb"] = "attr", img_size: int = 256, num_workers: int = 0, pin_memory = True, to_zero_one = True, train_set_size=-1):
        super().__init__(ds_id='celebA', batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, train_set_size=train_set_size)

        self.center_crop = center_crop
        self.target_type = target_type
        self.img_size = img_size
        self.to_zero_one = to_zero_one

        if type(target_type) == list:
            assert len(target_type) == 2
            self.train_ds = CelebACombined(
                root=DOWNLOAD_FOLDER,
                split="train",
                download=True,
                targets=target_type,
                transform=self._get_transform(True),
                target_transform=MinusOne() if self.target_type == "identity" else transforms.ToDtype(torch.float32),
            )
            self.test_ds = CelebACombined(
                root=DOWNLOAD_FOLDER,
                split="test",
                download=True,
                targets=target_type,
                transform=self._get_transform(False),
                target_transform=MinusOne() if self.target_type == "identity" else transforms.ToDtype(torch.float32),
            )
        else:
            self.train_ds = datasets.CelebA(
                root=DOWNLOAD_FOLDER,
                split="train",
                target_type=self.target_type,
                download=True,
                transform=self._get_transform(True),
                target_transform=MinusOne() if self.target_type == "identity" else transforms.ToDtype(torch.float32),
            )
            self.test_ds = datasets.CelebA(
                root=DOWNLOAD_FOLDER,
                split="test",
                target_type=self.target_type,
                download=True,
                transform=self._get_transform(False),
                target_transform=MinusOne() if self.target_type == "identity" else transforms.ToDtype(torch.float32),
            )

        self.options.update({
            "class_name": "DatasetCelebA", 
            "center_crop": center_crop,
            "target_type": target_type,
            "img_size": img_size,
            "to_zero_one": to_zero_one,

        })

    def input_channels(self):
        return 3
    
    def output_channels(self):
        if type(self.target_type) == list:
            return sum([OUTPUT_CHANNELS[target_type] for target_type in self.target_type])

        return OUTPUT_CHANNELS[self.target_type]

    def _get_test_dataset(self):
        return self.test_ds
    
    def _get_train_dataset(self):
        return self.train_ds
    
    def _get_transform(self, is_train):
        return_transform = []

        return_transform.append(transforms.ToTensor())
        if self.center_crop != 178:
            return_transform.append(transforms.CenterCrop((self.center_crop, self.center_crop)))
        if self.img_size != 256:
            return_transform.append(transforms.Resize(self.img_size, antialias=False))
        if not self.to_zero_one:
            return_transform.append(transforms.Normalize(
                [0.5 for _ in range(3)],
                [0.5 for _ in range(3)],
            ))

        return transforms.Compose(return_transform)

class CelebACombined(datasets.CelebA):
    def __init__(self, root: str, split, targets, transform = None, target_transform = None, download = False) -> None:
        super().__init__(root, split, targets, transform, target_transform, download)


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X, (y_a, y_b) = super().__getitem__(index)
        y = torch.cat([y_a, y_b], dim=0)
        return (X, y)

class MinusOne(torch.nn.Module):
    '''
    Shift targets to start with 0. Has to be a module since Lambda cannot be pickled
    '''
    def forward(self, x):
        return x - 1

def save_scaled_version(size):
    '''
    saves it to img_align_celeba_size - rename it to img_align_celeba to make pytorch use it.
    '''
    # print(DOWNLOAD_FOLDER)
    reshape = transforms.Compose([transforms.CenterCrop(178), transforms.Resize((size, size))])
    DOWNLOAD_FOLDER = repo_dir("datasets", "downloads")
    mkdir(join(DOWNLOAD_FOLDER, "celeba", f"img_align_celeba_{str(size)}"))
    for path in tqdm(listdir(join(DOWNLOAD_FOLDER, "celeba", "img_align_celeba"))):
        image = Image.open(join(DOWNLOAD_FOLDER, "celeba", "img_align_celeba", path))
        image = reshape(image)
        image.save(join(DOWNLOAD_FOLDER, "celeba", f"img_align_celeba_{str(size)}", path))

if __name__ == "__main__":
    save_scaled_version(256)