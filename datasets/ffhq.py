import json
import os
import pandas as pd
import torchvision
from tqdm import tqdm
import torch
from torch.utils.data import random_split
torchvision.disable_beta_transforms_warning()
from skimage import io
from torch.utils.data import Dataset

from misc_helpers.helpers import repo_dir
from os.path import join, isfile, isdir
from os import listdir, mkdir, rename, getcwd
from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms.v2 as transforms

from datasets.dataset_base import CustomDataset, DOWNLOAD_FOLDER

class DatasetFFHQ(CustomDataset):
    '''
    Data downloaded from https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq?select=00000.png

    if to_zero_one is true, images are in [0, 1] (default). Otherwise, they are shifted to [-1, 1]
    '''
    def __init__(self, batch_size: int, img_size: int = 128, num_workers: int = 0, pin_memory=False, to_zero_one=True, train_set_size=-1):
        super().__init__(ds_id='DatasetFFHQ', batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, train_set_size=train_set_size)

        self.img_size = img_size
        if img_size <= 128:
            self.source_img_size = 128
        elif img_size <= 256:
            self.source_img_size = 256
        else:
            self.source_img_size = 0

        self.options.update({
            "class_name": "DatasetFFHQ", 
            "img_size": img_size,
            "source_img_size": self.source_img_size,
            "to_zero_one": to_zero_one,
        })

        print(DOWNLOAD_FOLDER)
        
        self.dataset = self._base_dataset(to_zero_one)
        generator = torch.Generator().manual_seed(0)
        self.dataset_test, self.dataset_train = random_split(self.dataset, [0.05, 0.95], generator=generator)

    def _base_dataset(self, to_zero_one):
        return datasets.ImageFolder(
            root=join(DOWNLOAD_FOLDER, "ffhq", str(self.source_img_size)), 
            transform=self._get_transform(to_zero_one),
        )

    def _get_test_dataset(self):
        return self.dataset_test
    
    def _get_train_dataset(self):
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
    
class DatasetFFHQAttributes(DatasetFFHQ):
    def __init__(self, batch_size: int, img_size: int = 128, num_workers: int = 0, pin_memory=False, to_zero_one=True, train_set_size=-1, load_images=True):
        self.load_images = load_images
        super().__init__(batch_size, img_size, num_workers, pin_memory, to_zero_one, train_set_size)

        self.options.update({
            "class_name": "DatasetFFHQAttributes", 
            "load_images": load_images, 
        })

    def _base_dataset(self, to_zero_one):
        return _DatasetFFHQAttributes(self.img_size, self._get_transform(to_zero_one), self.load_images)

class _DatasetFFHQAttributes(Dataset):
    def __init__(self, img_size, transform, load_images=True):
        self.img_size = str(img_size)
        self.load_images = load_images
        self.transform = transform

        labels_raw = pd.read_csv(join(DOWNLOAD_FOLDER, "ffhq", "attributes.csv"), index_col=0)
        self.images = []
        self.labels = []
        print("DatasetFFHQ - Preparing Attribute Dataset")
        for i in tqdm(range(70000)):
            if i in labels_raw.index and os.path.isfile(join(DOWNLOAD_FOLDER, "ffhq", self.img_size, "0", f"{str(i).zfill(5)}.png")):
                if load_images:
                    self.images.append(transform(io.imread(join(DOWNLOAD_FOLDER, "ffhq", self.img_size, "0", f"{str(i).zfill(5)}.png"))))
                else:
                    self.images.append(join(DOWNLOAD_FOLDER, "ffhq", self.img_size, "0", f"{str(i).zfill(5)}.png"))
                self.labels.append(torch.tensor(labels_raw.loc[[i]].values[0], dtype=torch.float))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.load_images:
            return self.images[idx], self.labels[idx]
        return self.transform(io.imread(self.images[idx])), self.labels[idx]


def save_scaled_version(size):
    # print(DOWNLOAD_FOLDER)
    DOWNLOAD_FOLDER = repo_dir("datasets", "downloads")
    mkdir(join(DOWNLOAD_FOLDER, "ffhq", str(size)))
    for path in tqdm(listdir(join(DOWNLOAD_FOLDER, "ffhq", "org"))):
        image = Image.open(join(DOWNLOAD_FOLDER, "ffhq", "org", path))
        image = image.resize((size, size))
        image.save(join(DOWNLOAD_FOLDER, "ffhq", str(size), path))

def to_random_classes(folder):
    for index, filename in tqdm(enumerate(listdir(join(DOWNLOAD_FOLDER, "ffhq", folder)))):
        if isfile(join(DOWNLOAD_FOLDER, "ffhq", folder, filename)):
            if not isdir(join(DOWNLOAD_FOLDER, "ffhq", folder, str(index%10))):
                mkdir(join(DOWNLOAD_FOLDER, "ffhq", folder, str(index%10)))
            rename(join(DOWNLOAD_FOLDER, "ffhq", folder, filename), join(DOWNLOAD_FOLDER, "ffhq", folder, str(index%10), filename))

def undo_random_classes(folder):
    for i in range(10):
        for filename in tqdm(listdir(join(DOWNLOAD_FOLDER, "ffhq", "128", str(i)))):
            rename(join(DOWNLOAD_FOLDER, "ffhq", folder, str(i), filename), join(DOWNLOAD_FOLDER, "ffhq", folder, filename))

def json_to_csv_attributes():
    GLASSES = {"NoGlasses": 0, "ReadingGlasses": 1, "Sunglasses": 2, "SwimmingGoggles": 3}

    result = []
    bug_count = 0
    for file_name in tqdm(listdir(join(DOWNLOAD_FOLDER, "ffhq", "attributes_json"))):
        if isfile(join(DOWNLOAD_FOLDER, "ffhq", "attributes_json", file_name)):
            with open(join(DOWNLOAD_FOLDER, "ffhq", "attributes_json", file_name)) as file:
                img_data = json.load(file)
                if len(img_data) <= 0:
                    bug_count += 1
                    continue

                img_data = img_data[0]

            img_vector = [file_name.split(".")[0]]

            groups = [
                img_data["faceRectangle"], 
                img_data["faceAttributes"]["headPose"],
                img_data["faceAttributes"]["facialHair"],
                img_data["faceAttributes"]["emotion"],
                img_data["faceAttributes"]["makeup"],
                img_data["faceAttributes"]["occlusion"],
                ]
            for group in groups:
                img_vector.extend([float(i) for _, i in group.items()])

            single_floats = [
                img_data["faceAttributes"]["smile"],         
                img_data["faceAttributes"]["age"],         
                img_data["faceAttributes"]["blur"]["value"],         
                img_data["faceAttributes"]["exposure"]["value"],         
                img_data["faceAttributes"]["noise"]["value"],         
                img_data["faceAttributes"]["hair"]["bald"],         
                float(img_data["faceAttributes"]["hair"]["invisible"]),
                ]
            img_vector.extend(single_floats)

            img_vector.append(float(img_data["faceAttributes"]["gender"] == "female"))
            img_vector.append(GLASSES[img_data["faceAttributes"]["glasses"]])     

            result.append(img_vector)     

    df = pd.DataFrame(result)
    df = df.sort_values(by=0)
    print(bug_count)
    df.to_csv(join(DOWNLOAD_FOLDER, "ffhq", "attributes.csv"), index=False)

if __name__ == "__main__":
    save_scaled_version(256)
