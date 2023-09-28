from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from misc_helpers.helpers import repo_dir
import torch
import numpy as np


DOWNLOAD_FOLDER = repo_dir("datasets", "downloads")

class CustomDataset:
    '''
    Wrapper for datasets to add 
    * training/test dataloaders
    * distributed sampler
    * options dict for the logger
    * train_set_size: option to retrieve a smaller training set. The size is only applied to the dataset of get_train_dataloader (not _get_train_dataset)
    '''
    def __init__(self, ds_id: str, batch_size: int, num_workers=0, pin_memory=False, shuffle_dataloader=True, test_ds=None, train_ds=None, train_set_size=-1):
        self.ds_id = ds_id
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle_dataloader = shuffle_dataloader
        self.train_set_size = train_set_size

        self.test_ds = test_ds
        self.train_ds = train_ds
        self.test_dataloader = None
        self.train_dataloader = None

        self.training_sampler = None
        self.test_sampler = None

        self.options = {
            "ds_id": self.ds_id,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "shuffle": self.shuffle_dataloader,
            "test_ds": test_ds,
            "train_ds": train_ds,
        }

    def get_test_dataloader(self, num_replicas=1, rank=0, batch_size=None, shuffle=False):
        # batchsize not defined -> use the default
        if batch_size == None:
            batch_size = self.batch_size

        if self.test_dataloader != None and batch_size == self.batch_size:
            return self.test_dataloader
        
        print(f"CustomDataset - Creating test dataloader for batchsize {batch_size}")
        self.batch_size = batch_size
        
        dataset = self._get_test_dataset()

        if num_replicas == 1:
            self.test_dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=True,
            )
        else:
            if self.test_sampler == None:
                self.test_sampler = DistributedSampler(dataset, num_replicas=num_replicas, rank=rank, shuffle=False, drop_last=False)
            self.test_dataloader = DataLoader(
                dataset,
                sampler=self.test_sampler,
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=0,
                pin_memory=False,
                drop_last=True,
            )
        return self.test_dataloader

    def get_train_dataloader(self, num_replicas=1, rank=0, batch_size=None):
        # batchsize not defined -> use the default
        if batch_size == None:
            batch_size = self.batch_size

        if self.train_dataloader != None and batch_size == self.batch_size:
            return self.train_dataloader
        
        print(f"CustomDataset - Creating train dataloader for batchsize {batch_size}")
        self.batch_size = batch_size

        dataset = self._get_train_dataset()
        if self.train_set_size > -1:
            print(f"CustomDataset - Using a subset of {self.train_set_size} samples")
            train_indices = torch.from_numpy(np.random.choice(len(dataset), size=(self.train_set_size), replace=False)) 
            dataset = Subset(dataset=dataset, indices=train_indices)

        if num_replicas == 1:
            self.train_dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle_dataloader,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                drop_last=True,
                persistent_workers=self.num_workers > 1 # starting at 2 workers, the init at the start of the epoch usually makes the model freeze for a few seconds
            )
        else:
            if self.training_sampler == None:
                self.training_sampler = DistributedSampler(dataset, num_replicas=num_replicas, rank=rank, shuffle=True, drop_last=False)
            self.train_dataloader = DataLoader(
                dataset,
                sampler=self.training_sampler,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers * num_replicas,
                pin_memory=self.pin_memory,
                drop_last=True,
                persistent_workers=True, # more gpus - more effort to create the workers
            )
        return self.train_dataloader
        

    def input_channels(self):
        raise NotImplementedError()

    def output_channels(self):
        raise NotImplementedError()

    def _get_train_dataset(self) -> Dataset:
        if self.train_ds != None:
            return self.train_ds
        raise NotImplementedError()

    def _get_test_dataset(self) -> Dataset:
        if self.test_ds != None:
            return self.test_ds
        raise NotImplementedError()
    
    def to(self, device):
        print(f"To method is not implemented fo the current dataset ({self.ds_id})")


'''
Fix for slow epoch start with when using many processes, from here:
https://discuss.pytorch.org/t/enumerate-dataloader-slow/87778/3

replaced with "persistent_workers=True"
'''

class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)