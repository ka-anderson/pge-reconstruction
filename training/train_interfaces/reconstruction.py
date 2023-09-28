# import threading
import warnings
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
import torchvision.transforms as T
import torch
from tqdm import tqdm

from datasets.noise_generator import NoiseGenerator
from models.model_helpers import freeze_parameters
from models.simple_models import MLPEnsemble
from training.logger import Logger
from .train_interface_base import TrainInterface

class ReconstrInterfaceNoise(TrainInterface):
    '''
    Train a pre generator so that for a feauture vector, it creates the noise that, when fed to (fixed) G and E, recreates that vector.
    Expects a dataset (x, y) with x as the encoder output (feature vector) and y the noise that out lead to X.

    Note that we are iterating over the same data in each epoch, because that seemed to improve the result.

    version 2: removed some unused features. Old version can be found in experiment 02_0
    '''
    def __init__(self, optimizer: torch.optim.Optimizer, generator: nn.Module, pre_gen: nn.Module, loss_fn) -> None:
        self.generator = freeze_parameters(generator).eval()

        self.pre_gen = pre_gen
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.options = {
            'class_name': 'ReconstrInterfaceNoise',
            "class_group": "ReconstrInterfaceP",
            'optimizer': optimizer,
            'generator': generator,
            'pre_gen': pre_gen,
            'loss_fn': loss_fn,
            'version': 2
        }
        
    def call_for_all_models(self, func):
        self.generator = func(self.generator) 
        self.pre_gen = func(self.pre_gen)

    def call_for_all_learning_models(self, func):
        self.pre_gen = func(self.pre_gen)
    
    def get_model_out(self, input):
        noise = self.pre_gen(input)
        return self.generator(noise)
    
    def get_learning_models(self):
        return {"model": self.pre_gen}
    
    def train(self, dataloader, logger:Logger):
        self.pre_gen.train()
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            pred = self.pre_gen(X)
            loss = self.loss_fn(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.print_substep(logger, batch * len(X), size, {"loss": loss})

        return {"loss": loss.item()}
    
class ReconstrInterfaceNoiseEncoder(ReconstrInterfaceNoise):
    '''
    Noise based training, which does not require a dataset, but creates the dataset on the go, using a noise generator and the encoder.
    '''
    def __init__(self, optimizer: torch.optim.Optimizer, generator: nn.Module, pre_gen: nn.Module, loss_fn, encoder: nn.Module, noise_gen: NoiseGenerator, batches_per_epoch: int) -> None:
        super().__init__(optimizer, generator, pre_gen, loss_fn)

        self.encoder = freeze_parameters(encoder).eval()
        self.noise_gen = noise_gen
        self.batches_per_epoch = batches_per_epoch

        self.options.update({
            'class_name': 'ReconstrInterfaceNoiseEncoder',
            "class_group": "ReconstrInterfacePImage",
            "enc": encoder,
            "noise_gen": noise_gen,
            "batches_per_epoch": batches_per_epoch,
        })
    
    def call_for_all_models(self, func):
        super().call_for_all_models(func)
        self.encoder = func(self.encoder)
        self.noise_gen = func(self.noise_gen)

    def train(self, dataloader, logger: Logger, epoch):
        self.pre_gen.train()
        for i in range(self.batches_per_epoch):
            with torch.no_grad():
                noise = self.noise_gen.get_noise()
                img = self.generator(noise)
                f = self.encoder(img).detach()

            rec_noise = self.pre_gen(f)
            loss = self.loss_fn(rec_noise, noise)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.print_substep(logger, i * len(noise), self.batches_per_epoch, {"loss": loss})

        return {"loss": loss.item()}


class ReconstrInterfaceImage(TrainInterface):
    '''
    Train a pre generator so that for a feature vector, it creates the noise that, when fed to (fixed) G and E, recreates that vector.

    Different than the original, ReconstrInterfaceP, we are not working with a fixed dataset made by feeding noise to G->E to get the matching f,
    but we are iterating over an image dataset, and 

    1. retrieving f=E(I), detach it (assuming E to be blackbox)
    2. calculating a recreated image I' = G(P(f))
    3. updating P on the loss between I and I'

    ReconstrInterfaceP was originally made to compare the loss between n, ReconstrInterfacePImage is intended for the loss between I. More detail in exp 3.3.

    Training can be either done with any image dataset, given an enocder, or without an encoder, using a dataset with tuples (x,y) where x is the image and y the feature vector (not that the image/the "target" comes first)
    '''

    def __init__(self, optimizer, loss_fn, generator, pre_gen, encoder=None, loss_image_size=None, noise_loss_fn=None) -> None:
        super().__init__(optimizer, loss_fn)

        self.generator = freeze_parameters(generator)
        self.generator.eval()
        if encoder != None:
            self.encoder = freeze_parameters(encoder)
            self.encoder.eval()
        else:
            self.encoder = None

        self.pre_gen = pre_gen
        self.loss_image_size = loss_image_size
        if loss_image_size:
            self.resize = T.Resize((loss_image_size, loss_image_size), antialias=None)
        self.noise_loss_fn = noise_loss_fn

        self.options.update({
            "class_name": "ReconstrInterfacePImage",
            "class_group": "ReconstrInterfacePImage",
            "generator": generator,
            "pre_gen": pre_gen,
            "enc": encoder,
            "loss_image_size": loss_image_size,
            "noise_loss_fn": noise_loss_fn,
        })

    def call_for_all_models(self, func):
        self.generator = func(self.generator) 
        self.pre_gen = func(self.pre_gen)
        if self.encoder != None:
            self.encoder = func(self.encoder)
        self.loss_fn = func(self.loss_fn) # loss is usually a model, like facenet.
    
    def call_for_all_learning_models(self, func):
        self.pre_gen = func(self.pre_gen)
    
    def get_learning_models(self):
        return {"model": self.pre_gen}
    
    def train(self, dataloader, logger: Logger, _) -> dict:
        self.pre_gen.train()
        size = len(dataloader.dataset)
        for batch, (images, features) in enumerate(dataloader):
            images = images.to(self.device)

            if self.encoder != None:
                features = self.encoder(images).detach()
            else:
                features = features.to(self.device) # assuming the "target" of the dataset are the features

            pred_n = self.pre_gen(features)

            if self.noise_loss_fn != None:
                noise_loss = self.noise_loss_fn(pred_n)
            else:
                noise_loss = 0

            pred_images = self.generator(pred_n)

            if self.loss_image_size:
                pred_images = self.resize(pred_images)
                images = self.resize(images)
                
            image_loss = self.loss_fn(pred_images, images) # loss function could return multiple values, the first as them main loss, everything else als additional info
            if isinstance(image_loss, tuple):
                total_loss = image_loss[0]
                image_loss = [loss.item() for loss in image_loss]
            else:
                total_loss = image_loss

            total_loss = total_loss + noise_loss / 100
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            self.print_substep(logger, batch * len(images), size, {"image_loss": image_loss, "noise_loss": noise_loss})

        return {"image_loss": image_loss if isinstance(image_loss, list) else image_loss.item(), "noise_loss": noise_loss.item()}
    
class ReconstrInterfaceImageFromNoise(ReconstrInterfaceImage):
    '''
    Same as ReconstrInterfaceImage, but instead of iterating over a dataset, we generating random noise, so that the target images are generated by G

    batches_per_epoch only determines how often the values are logged. Every single batch contains freshly generated noise.
    '''
    def __init__(self, optimizer, loss_fn, generator, pre_gen, encoder, noise_gen: NoiseGenerator, batches_per_epoch) -> None:
        super().__init__(optimizer, loss_fn, generator, pre_gen, encoder, None)

        self.noise_gen = noise_gen
        self.batches_per_epoch = batches_per_epoch

        self.options.update({
            "class_name": "ReconstrInterfaceImageFromNoise",
            "noise_gen": noise_gen,
            "batches_per_epoch": batches_per_epoch,
        })

    def call_for_all_models(self, func):
        super().call_for_all_models(func)
        self.noise_gen = func(self.noise_gen)

    def train(self, dataloader, logger: Logger, _) -> dict:
        self.pre_gen.train()
        for i in range(self.batches_per_epoch):
            with torch.no_grad():
                noise = self.noise_gen.get_noise()
                images = self.generator(noise)
                f = self.encoder(images).detach()

            pred_n = self.pre_gen(f)
            pred_images = self.generator(pred_n)

            if self.loss_image_size:
                pred_images = self.resize(pred_images)
                images = self.resize(images)
                
            loss = self.loss_fn(pred_images, images) # loss function could return multiple values, the first as them main loss, everything else als additional info
            if isinstance(loss, tuple):
                main_loss = loss[0]
                loss = [loss.item() for loss in loss]
            else:
                main_loss = loss

            self.optimizer.zero_grad()
            main_loss.backward()
            self.optimizer.step()

            self.print_substep(logger, i * len(images), self.batches_per_epoch * images.shape[0], {"loss": loss})

        return {"loss": loss if isinstance(loss, list) else loss.item()}



class ReconstructionInterfaceImageEnsemble(TrainInterface):
    '''
    An ensemble interface that was optimized to work with DDP, since DDP wrapping does not mix well with the MLP-Ensemble. 
    We are using one MLP for each loss function, so loading a model that was trained for a different loss composition DOES NOT WORK.

    For the pre_gen ensemble:
    * Expects it ensemble to contain exactly one MLP for each loss function,
    * mlp i is updated accodring to the i-th loss function
    * it is left to the ensemble to create the stacked latent vector (using the expected composition of MLP outputs) 
    '''
    def __init__(self, generator, encoder, pre_gen: MLPEnsemble, loss_functions: dict, optimizer_base, lr, fuzzy_lambda) -> None:
        super().__init__(None, None, None)

        num_loss_functions = len(loss_functions.keys())
        assert pre_gen.num_mlps >= num_loss_functions
        if pre_gen.num_mlps < num_loss_functions:
            warnings.warn(f"{num_loss_functions} for {pre_gen.num_mlps} mlps, {num_loss_functions - pre_gen.num_mlps} mlps will remain frozen.")
    
        self.fuzzy_lambda = fuzzy_lambda

        self.generator = freeze_parameters(generator).eval()
        self.encoder = freeze_parameters(encoder).eval()

        self.pre_gen = pre_gen
        self.model_groups = {}
        for i, (key, loss_fn) in enumerate(loss_functions.items()):
            p = pre_gen.get_module(i)
            self.model_groups[key] = {
                "p": p,
                "opt": optimizer_base(p.parameters(), lr=lr),
                "loss_fn": loss_fn,
            }
        
        self.options.update({
            "class_name": "ReconstructionInterfacePEnsemble",
            "class_group": "ReconstrInterfacePImage",
            "optimizer_base": optimizer_base,
            "lr": lr,
            "fuzzy_lambda": fuzzy_lambda,

            "generator": generator,
            "encoder": encoder,
            "pre_gen": pre_gen,

            "loss_functions": loss_functions,
            "model_groups": self.model_groups,
        })
    
    def call_for_all_models(self, func):
        self.generator = func(self.generator)
        self.encoder = func(self.encoder)
        self.pre_gen = func(self.pre_gen)
    
    def get_learning_models(self):
        return {"model": self.pre_gen}

    def to(self, device, distributed=False, find_unused_parameters=False):
        print(f"ReconstructionInterfaceImageEnsemble - moving all models to device {device} (distributed: {distributed})")
        def move(model):
            model = model.to(device)
            return model
        
        self.device = device
        self.call_for_all_models(move)

        # wrap each mlp into its own DDP
        if distributed:
            for key, _ in self.model_groups.items():
                self.model_groups[key]["p"] = DDP(self.model_groups[key]["p"], device_ids=[device], output_device=device, find_unused_parameters=find_unused_parameters)
    

    def train(self, dataloader, logger: Logger, epoch: int) -> dict:
        size = len(dataloader.dataset)
        final_loss_fn_index = len(self.model_groups) - 1

        for batch, (images, _) in enumerate(dataloader):
            images = images.to(self.device)

            f = self.encoder(images).detach()

            # requires_grad for the calculation of the image
            freeze_parameters(self.pre_gen, requires_grad=True)

            n_list = self.pre_gen(f)
            latent_list = [self.generator.model.mapping(n, None)[:, 0, :] for n in n_list]
            latent = torch.stack(latent_list, dim=1)

            if self.fuzzy_lambda != 0:
                latent = (1 - self.fuzzy_lambda * latent) + self.fuzzy_lambda * torch.mean(latent, dim=1, keepdim=True)

            pred_images = self.generator.model.synthesis(latent)
            if hasattr(self.generator, "resize"):
                pred_images = self.generator.resize(pred_images)

            loss_out = {}
            # freeze all p, to unfreeze them one by one. Might not be required, but can't hurt. I hope.
            freeze_parameters(self.pre_gen, requires_grad=False)
            for i, (key, values) in enumerate(self.model_groups.items()):
                freeze_parameters(values["p"], requires_grad=True)
                values["opt"].zero_grad()
                loss = values["loss_fn"](pred_images, images)
                loss.backward(retain_graph = (i < final_loss_fn_index))
                values["opt"].step()
                freeze_parameters(values["p"], requires_grad=False)

                loss_out[key] = loss.item()
            
            self.print_substep(logger, batch * len(images), size, loss_out)

        return loss_out