import torchvision
from tqdm import tqdm
torchvision.disable_beta_transforms_warning()

from os.path import join
from os import listdir, mkdir
from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms.v2 as transforms

from datasets.dataset_base import CustomDataset, DOWNLOAD_FOLDER

class DatasetCelebAHQ(CustomDataset):
    '''
    Data downloaded from https://www.kaggle.com/datasets/lamsimon/celebahq
    * an image size other than 0 results in the images being rescaled using transforms.Resize.
    '''
    def __init__(self, batch_size: int, img_size: int = 1024, num_workers: int = 0, pin_memory=False, to_zero_one=True):
        super().__init__(ds_id='CelebAHQ', batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

        self.img_size = img_size
        self.to_zero_one = to_zero_one

        self.source_img_size = 128 if img_size <= 128 else 1024

        self.options.update({
            "class_name": "DatasetCelebAHQ", 
            "img_size": img_size,
            "source_img_size": self.source_img_size,
            "to_zero_one": to_zero_one,
        })

        self.dataset_train = datasets.ImageFolder(root=join(DOWNLOAD_FOLDER, "celeba_hq", str(self.source_img_size), "train"), transform=self._get_transform(True))
        self.dataset_test = datasets.ImageFolder(root=join(DOWNLOAD_FOLDER, "celeba_hq", str(self.source_img_size), "val"), transform=self._get_transform(False))


    def _get_test_dataset(self):
        return self.dataset_test
    
    def _get_train_dataset(self):
        return self.dataset_train

    def _get_transform(self, is_train):
        return_transform = []

        if self.img_size != self.source_img_size:
            return_transform.append(transforms.Resize((self.img_size, self.img_size), antialias=None))

        return_transform.append(transforms.ToTensor())

        if is_train:
            return_transform.append(transforms.RandomHorizontalFlip(p=0.5))

        if not self.to_zero_one:
            return_transform.append(transforms.Normalize(
                    [0.5 for _ in range(3)],
                    [0.5 for _ in range(3)],
                ))

        return transforms.Compose(return_transform)

def save_scaled_version(size):
    mkdir(join(DOWNLOAD_FOLDER, "celeba_hq", str(size)))
    for path_0 in listdir(join(DOWNLOAD_FOLDER, "celeba_hq", "1024")):
        mkdir(join(DOWNLOAD_FOLDER, "celeba_hq", str(size), path_0))
        for path_1 in listdir(join(DOWNLOAD_FOLDER, "celeba_hq", "1024", path_0)):
            mkdir(join(DOWNLOAD_FOLDER, "celeba_hq", str(size), path_0, path_1))
            for path_image in tqdm(listdir(join(DOWNLOAD_FOLDER, "celeba_hq", "1024", path_0, path_1))):
                image = Image.open(join(DOWNLOAD_FOLDER, "celeba_hq", "1024", path_0, path_1, path_image))
                image = image.resize((size, size))
                image.save(join(DOWNLOAD_FOLDER, "celeba_hq", str(size), path_0, path_1, path_image))

if __name__ == "__main__":
    save_scaled_version(128)