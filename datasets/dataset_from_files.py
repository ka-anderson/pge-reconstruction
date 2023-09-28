import torch
from torch.utils.data import Dataset

class NoiseFeatureDS(Dataset):
    def __init__(self, features, noise, images=None, target_as_image=False) -> None:
        '''
        default: return samples (features, noise) with: Generator(noise) = I, T(I) = features

        target_as_image: return samples (features, image) with: Generator(noise) = Image 
        (used to evaluate the training loss: is the image created from the learned "noise" similar to the image created from the target noise?)
        '''
        self.target_as_image = target_as_image
        self.features = features
        self.noise = noise
        self.images = images

        self.options = {
            "classname": "NoiseFeatureDS",
            "target_as_image": target_as_image,
        }

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        if self.target_as_image:
            return self.features[index], self.images[index]
        else:
            return self.features[index], self.noise[index]

class ImageFeatureDS(Dataset):
    def __init__(self, features, image_dataset) -> None:
        self.image_dataset = image_dataset
        self.features = features

        self.options = {
            "classname": "ImageFeatureDS",
            "image_dataset": image_dataset,
        }

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.image_dataset.__getitem__(index)[0], torch.tensor(self.features[index])