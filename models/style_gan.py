import torch
import torch.nn as nn
import torchvision.transforms as T
from misc_helpers.helpers import repo_dir
from models.model_helpers import IdentityModule
from models.simple_models import MLP

import pickle


class FinetunedStyleDiscriminator(nn.Module):
    '''
    Load a pretrained stylegan discriminator (default ffhq 256), (optionally) attach a linear finetuning layer and, if required, a resizing step
    '''

    def __init__(self, use_finetuning_layer, img_size=64, pretrained_model_path="stylegan_ffhq_256", eval=True):
        super(FinetunedStyleDiscriminator, self).__init__()

        img_size = int(img_size)
        if type(use_finetuning_layer) == "str":
            use_finetuning_layer = use_finetuning_layer == "True"

        with open(repo_dir("models", "pretrained", f"{pretrained_model_path}.pkl"), 'rb') as f:
            self.model = pickle.load(f)["D"]  # torch.nn.Module

        self.img_size = img_size
        if self.img_size != 256:
            self.resize = T.Resize((256, 256), antialias=None)

        self.options = {
            "pretrained_model_path": pretrained_model_path,
            "img_size": img_size,
            "use_finetuning_layer": use_finetuning_layer,
            "eval": eval,
        }
        if use_finetuning_layer:
            self.finetuning_layer = MLP(in_channels=3, out_channels=3, hidden_channels=[], in_h=self.img_size, in_w=self.img_size, out_h=self.img_size, out_w=self.img_size)
            if eval:
                self.finetuning_layer.eval()
            self.options.update({"finetuning_layer": self.finetuning_layer,})
        
        if eval:
            self.model.eval()

    def to(self, device):
        if hasattr(self, "finetuning_layer"):
            self.finetuning_layer = self.finetuning_layer.to(device)
        self.model = self.model.to(device)
        return self

    def forward(self, x):
        if hasattr(self, "resize"):
            x = self.resize(x)

        if hasattr(self, "finetuning_layer"):
            x = self.finetuning_layer(x)
            
        x = self.model(x, None)
        return x


class FinetunedStyleGenerator(nn.Module):
    '''
    Load a pretrained stylegan generator (default ffhq 256). Optionally:
        * attach a linear finetuning layer (in_channels != pretrained_in_channels)
        * end with a resizing step (img_size != pretrained_img_size)
    '''
    def __init__(self, in_channels=512, pretrained_in_channels=512, img_size=64, pretrained_model_path="stylegan_ffhq_256", pretrained_img_size=256, styleGAN_seed=0):
        super(FinetunedStyleGenerator, self).__init__()
        
        in_channels = int(in_channels)
        pretrained_in_channels = int(pretrained_in_channels)
        img_size = int(img_size)
        pretrained_img_size = int(pretrained_img_size)

        last_rng_state = torch.get_rng_state()
        torch.manual_seed(styleGAN_seed)
        with open(repo_dir("models", "pretrained", f"{pretrained_model_path}.pkl"), 'rb') as f:
            self.model = pickle.load(f)["G_ema"]  # torch.nn.Module
        torch.set_rng_state(last_rng_state)

        if in_channels != pretrained_in_channels:
            self.finetuning_layer = MLP(in_channels=in_channels, out_channels=512, hidden_channels=[], output_vector=True)
        else: 
            self.finetuning_layer = IdentityModule() # finetuning layer is used by ensemble training interface
        
        if img_size != pretrained_img_size:
            self.resize = T.Resize((img_size, img_size), antialias=None)

        self.options = {
            "class_name": "FinetunedStyleGenerator",
            "pretrained_model_path": pretrained_model_path,
            "in_channels": in_channels,
            "pretrained_in_channels": pretrained_in_channels,
            "img_size": img_size,
            "pretrained_img_size": pretrained_img_size,
            "finetuning_layer": self.finetuning_layer,
            "styleGAN_seed": styleGAN_seed,
        }

    def to(self, device):
        self.device = device
        self.finetuning_layer = self.finetuning_layer.to(device)
        self.model = self.model.to(device)
        return self

    def mapping(self, x):
        return self.model.mapping(x, None)
    
    def synthesis(self, x):
        return self.model.synthesis(x, noise_mode="const")

    def forward(self, x):
        x = self.finetuning_layer(x)
        x = self.mapping(x)
        x = self.synthesis(x)

        if hasattr(self, "resize"):
            x = self.resize(x)

        return x