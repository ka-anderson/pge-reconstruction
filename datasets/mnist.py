import torchvision.datasets as datasets
from torchvision import transforms

from .dataset_base import CustomDataset, DOWNLOAD_FOLDER

# zero mean and unit sd?
# https://stackoverflow.com/questions/63746182/correct-way-of-normalizing-and-scaling-the-mnist-dataset
MNIST_NORM = transforms.Normalize((0.1307,), (0.3081,))
IMAGE_NORM = transforms.Normalize([0.5], [0.5])

class DatasetMNIST(CustomDataset):
    '''
    * an image size other than 28 results in the images being rescaled using transforms.Resize
    * normalization:
        * "none": default [0, 1] range.
        * "mean0": transforms.Normalize((0.1307,), (0.3081,)) to get mean 0, variance 1
        * "05": transforms.Normalize([0.5], [0.5]) since that somehow works best for some GANs
    '''
    def __init__(self, normalization: str, batch_size: int, img_size: int = 28, num_workers: int = 0):
        super().__init__(ds_id='mnist', batch_size=batch_size, num_workers=num_workers)

        assert normalization in ["none", "mean0", "05"], f'{normalization} is no normalization mode'

        self.normalization = normalization
        self.img_size = img_size
        self.options.update({
            "class_name": "DatasetMNIST", 
            "normalization": normalization,
            "img_size": img_size,
        })

    def input_channels(self):
        return 1 # img size 28*28
    
    def output_channels(self):
        return 10

    def _get_test_dataset(self):
        return datasets.MNIST(
            root=DOWNLOAD_FOLDER,
            train=False,
            download=True,
            transform=self._get_transform(False),
        )
    
    def _get_train_dataset(self):
        return datasets.MNIST(
            root=DOWNLOAD_FOLDER,
            train=True,
            download=True,
            transform=self._get_transform(True)
        )

    def _get_transform(self, is_train):
        return_transform = []

        # return_transform.append(transforms.Lambda(lambda x: x.convert('RGB')))
        return_transform.append(transforms.ToTensor())
        if self.img_size != 28:
            return_transform.append(transforms.Resize((self.img_size, self.img_size), antialias=False))

        if self.normalization == "mean0":
            return_transform.append(MNIST_NORM)
        elif self.normalization == "05":
            return_transform.append(IMAGE_NORM)

        return transforms.Compose(return_transform)