import copy
import torch
from torch import nn
from torchvision.transforms.functional import crop
from .model_helpers import DebugLogger, FrechetModule, IdentityModule
from torchvision.models import mobilenet_v2, resnet18, ResNet18_Weights

ACTIVATIONS = {
    "relu": nn.ReLU(),
    "lrelu": nn.LeakyReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "none": IdentityModule(),
}

class CustomMobilenetV2(FrechetModule):
    '''
    version of mobilenetV2 with customizable in- and output dimension
    Does not extend MobileNetV2 class, this somehow did not work
    '''
    def __init__(self, in_dim=3, out_dim=1000) -> None:
        super(CustomMobilenetV2, self).__init__()

        self.model = mobilenet_v2()
        in_dim = int(in_dim)
        out_dim = int(out_dim)
        self.out_dim = out_dim

        self.used_for_fid = False

        # from https://www.kaggle.com/code/zfturbo/mnist-with-mobilenet-pytorch-gpu:
        if in_dim != 3:
            self.model.features[0][0] = torch.nn.Conv2d(in_channels=in_dim, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        if out_dim != 1000:
            self.model.classifier[1] = torch.nn.Linear(in_features=self.model.classifier[1].in_features, out_features=out_dim)

        self.options = {
            "classname": "CustomMobilenetV2",
            "in_dim": in_dim,
            "out_dim": out_dim,
        }

    def forward(self, x):
        # fid dummy image is uint8 and much too large, but only used to get the output dimension
        if x.dtype == torch.uint8:
            assert self.used_for_fid
            print("CustomMobilenetV2 is returning dummy output!")
            return torch.zeros((x.shape[0], self.out_dim))
        
        x = self.model(x)

        if self.used_for_fid:
            x = x.view(x.shape[0], x.shape[1])
        
        return x

class CustomResNet18(nn.Module):
    def __init__(self, weights=ResNet18_Weights.DEFAULT, in_dim=3, out_dim=1000) -> None:
        super(CustomResNet18, self).__init__()

        self.model = resnet18(weights=weights)

        in_dim = int(in_dim)
        out_dim = int(out_dim)

        if in_dim != 3:
            self.model.conv1 = torch.nn.Conv2d(in_channels=in_dim, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if out_dim != 1000:
            self.model.fc = torch.nn.Linear(in_features=self.model.fc.in_features, out_features=out_dim)

        self.options = {
            "classname": "CustomResNet18",
            "in_dim": in_dim,
            "out_dim": out_dim,
        }

    def forward(self, x):
        return self.model(x)

class LeNet5(FrechetModule):
    '''
    https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320

    But using additional padding in the first conv
    Is also used to evaluate the FID on small greyscale images
    '''
    def __init__(self, out_dim, in_dim=1, input_img_size=28):
        super(LeNet5, self).__init__()

        out_dim = int(out_dim)
        in_dim = int(in_dim)
        input_img_size = int(input_img_size)
        self.used_for_fid = False
        
        self.feature_extractor = nn.Sequential(
            # in_dim, img_size       
            nn.Conv2d(in_channels=in_dim, out_channels=6, kernel_size=5, stride=1, padding=2), # -> 6, img_size
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2), # -> 6, img_size/2
            # 6, img_size/2    
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1), # -> 16, (img_size/2) - 4
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2), # -> 16, ((img_size/2) - 4)/2
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1), # -> 120, (((img_size/2) - 4)/2) - 4 = (img_size-24)/4
            nn.Tanh()
        )

        final_img_size = int((input_img_size-24)/4)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120 * final_img_size * final_img_size, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=out_dim),
        )

        self.out_dim = out_dim

        self.options = {
            "classname": "LeNet5",
            "out_dim": out_dim,
            "in_dim": in_dim,
            "input_img_size": input_img_size,
        }

    def forward(self, x):
        # fid dummy image is uint8 and much too large, but only used to get the output dimension
        if x.dtype == torch.uint8:
            assert self.used_for_fid
            print("LeNET5 is returning dummy output!")
            return torch.zeros((x.shape[0], self.out_dim))

        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        if self.used_for_fid:
            x = x.view(x.shape[0], x.shape[1])

        return x

    
class MLP(nn.Module):
    '''
    version 2: added leaky relu option. Default activation is still relu, as it was for version 1
    '''
    def __init__(self, out_channels: int, in_channels: int, hidden_channels:"list[int]", in_w: int = 1, in_h: int = 1, out_w: int = 1, out_h: int = 1, output_vector=False, activation="relu", final_activation="none") -> None:
        '''
        activation and final_activation must be "sigmoid", "tanh", "relu" or "lrelu" (leaky relu with slope 0.2), or "none"
        '''
        super(MLP, self).__init__()

        out_channels = int(out_channels)
        in_channels = int(in_channels)
        in_w = int(in_w)
        in_h = int(in_h)
        out_w = int(out_w)
        out_h = int(out_h)
        if type(output_vector) == str:
            output_vector = output_vector == "True"

        if type(hidden_channels) == str:
            if hidden_channels == "[]":
                hidden_channels = []
            else:
                hidden_channels = [int(a) for a in hidden_channels.replace("[", "").replace("]", "").split(", ")]

        self.output_vector = output_vector
        self.out_h = out_h
        self.out_w = out_w
        self.out_channels = out_channels

        channels = [in_channels * in_h * in_w]
        channels.extend(hidden_channels)
        channels.append(out_channels * out_h * out_w)

        self.layers = nn.Sequential()
        for i in range(len(channels) - 1):
            self.layers.add_module(f"layer_{i}", nn.Linear(channels[i], channels[i+1]))
            if i < (len(channels) - 2): # not the output layer
                if activation not in ACTIVATIONS:
                    raise Exception(f"Unknown activation name: {activation}.")
                
                self.layers.add_module(f"{activation}_{i}", ACTIVATIONS[activation])
        self.layers.add_module("final_activation", ACTIVATIONS[final_activation])

        self.options = {
            "classname": "MLP",
            "out_channels": out_channels,
            "in_channels": in_channels,
            "in_w": in_w,
            "in_h": in_h,
            "out_w": out_w,
            "out_h": out_h,
            "output_vector": output_vector,
            "hidden_channels": hidden_channels,
            "activation": activation,
            "final_activation": final_activation,
            "version": 2,
        }

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.layers(x)
        if not self.output_vector:
            x = x.view((x.shape[0], self.out_channels, self.out_h, self.out_w))
        else:
            x = x.view((x.shape[0], self.out_channels))

        return x
    

class MLPEnsemble(nn.Module):
    '''
    returns a list with outputs of different MLPs. MLPs are initialized as clones.

    * v2: added to option to implement a custom mapping from mlp to output vector: 
        [i_0, i_1, ... i_n] where i_j means that the output vector position j is the output of mlp with index i_j
        e.g. [0, 0, 0, 1] makes output item 0,1,2 the output of mlp number 0, and the final output (4) item the output of mlp number 2.
    '''
    def __init__(self, num_mlps, out_channels: int, in_channels: int, hidden_channels:"list[int]", in_w: int = 1, in_h: int = 1, out_w: int = 1, out_h: int = 1, activation="relu", final_activation="none", init_seed=0, output_mapping=None) -> None:
        super(MLPEnsemble, self).__init__()
        self.num_mlps = int(num_mlps)
        if output_mapping == None:
            output_mapping = [i for i in range(num_mlps)]
        self.output_mapping = output_mapping

        last_rng_state = torch.get_rng_state()
        torch.manual_seed(init_seed)
        base_mlp = MLP(out_channels, in_channels, hidden_channels, in_w, in_h, out_w, out_h, output_vector=True, activation=activation, final_activation=final_activation)
        self.components = nn.ModuleList([copy.deepcopy(base_mlp) for _ in range(self.num_mlps)])
        torch.set_rng_state(last_rng_state)

        self.options = {
            "classname": "MLPEnsemble",
            "num_mlps": num_mlps,
            "output_mapping": self.output_mapping,

            "out_channels": out_channels,
            "in_channels": in_channels,
            "in_w": in_w,
            "in_h": in_h,
            "out_w": out_w,
            "out_h": out_h,
            "hidden_channels": hidden_channels,
            "activation": activation,
            "final_activation": final_activation,
            "init_seed": init_seed,
        }

    def copy_mlp_zero(self):
        '''
        clones the weights of module 0 into all other modules
        '''
        base_mlp = copy.deepcopy(self.components[0])
        self.components = nn.ModuleList([copy.deepcopy(base_mlp) for _ in range(self.num_mlps)])
    
    def get_module(self, index):
        return self.components[index]

    def forward(self, x):
        mlp_out = []
        for mlp in self.components:
            mlp_out.append(mlp(x))

        stacked_out = []
        for mlp_index in self.output_mapping:
            stacked_out.append(mlp_out[mlp_index])

        return stacked_out
    
class ConvStack(FrechetModule):
    '''
    * Convolutions from in_channels to the final hidden_channel, each followed by activation and BN
    * when using regular convs: 
        * avg pooling of each feature map
        * linear layer from final hidden_channel to out_channels*out_h*out_w
        * reshape to an image shaped out_channels x out_h x out_w
    '''
    def __init__(self, channels:"list[int]", k, s, p, out_h: int = 1, out_w: int = 1, activation="relu"):
        super(ConvStack, self).__init__()

        out_h, out_w, k, s, p = int(out_h), int(out_w), int(k), int(s), int(p)
        if type(channels) == str:
            if channels == "[]":
                channels = []
            else:
                channels = [int(a) for a in channels.replace("[", "").replace("]", "").split(", ")]

        self.out_h = out_h
        self.out_w = out_w
        self.channels = channels

        if activation not in ACTIVATIONS:
            raise Exception(f"Unknown activation name: {activation}.")

        self.layers = nn.Sequential()
        self.setup_layers(activation, k, s, p)
        self.used_for_fid = False

        self.options = {
            "classname": "ConvStack",
            "channels": channels,
            "k": k,
            "s": s,
            "p": p,
            "out_w": out_w,
            "out_h": out_h,
            "activation": activation,
        }
    
    def setup_layers(self, activation, k, s, p):
        for i in range(len(self.channels) - 2):
            self.layers.add_module(f"conv_{i}", nn.Conv2d(in_channels=self.channels[i], out_channels=self.channels[i+1], kernel_size=k, stride=s, padding=p))
            if i < (len(self.channels) - 2):
                self.layers.add_module(f"bn_{i}", nn.BatchNorm2d(self.channels[i+1]))
                self.layers.add_module(f"{activation}_{i}", ACTIVATIONS[activation])

        self.linear = nn.Linear(self.channels[-2], self.channels[-1] * self.out_h * self.out_w)


    def forward(self, x):
        # fid dummy image is uint8 and much too large, but only used to get the output dimension
        if x.dtype == torch.uint8:
            assert self.used_for_fid
            print("ConvStack is returning dummy output!")
            return torch.zeros((x.shape[0], self.channels[-1] * self.out_h * self.out_w))

        x = self.layers(x)

        x = x.mean([-2, -1])
        x = self.linear(x)
        x = x.view((x.shape[0], self.channels[-1], self.out_h, self.out_w))

        if self.used_for_fid:
            x = x.view(x.shape[0], self.channels[-1] * self.out_h * self.out_w)

        return x
    
class TransposedConvStack(ConvStack):
    '''
    resolution starts at 4x4.
    for k=4, s=2, p=1, the image size is doubled in each step
    '''

    def __init__(self, channels: "list[int]", k, s, p, out_h: int = 1, out_w: int = 1, activation="relu"):
        super().__init__(channels, k, s, p, out_h, out_w, activation)
        self.options["classname"] = "TransposedConvStack"

    def setup_layers(self, activation, k, s, p):
        self.linear = nn.Linear(self.channels[0], self.channels[1] * 16)

        # self.layers.add_module(f"conv_{i}", nn.ConvTranspose2d(in_channels=self.channels[i], out_channels=self.channels[i+1], kernel_size=k, stride=s, padding=p))

        for i in range(1, len(self.channels) - 1):
            self.layers.add_module(f"conv_{i}", nn.ConvTranspose2d(in_channels=self.channels[i], out_channels=self.channels[i+1], kernel_size=k, stride=s, padding=p))
            # self.layers.add_module(f"debug_{i}", DebugLogger())
            if i < (len(self.channels) - 2):
                self.layers.add_module(f"bn_{i}", nn.BatchNorm2d(self.channels[i+1]))
                self.layers.add_module(f"{activation}_{i}", ACTIVATIONS[activation])
        

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        x = x.view(x.shape[0], self.channels[1], 4, 4)

        x = self.layers(x)
        return x


class CroppedEncoder(torch.nn.Module):
    '''
    Crop the input before feeding it to the encoder. crop ist a list with [top, left, width, height]
    '''
    def __init__(self, encoder, crop) -> None:
        super(CroppedEncoder, self).__init__()
        self.crop = crop
        self.encoder = encoder

        self.options = {
            "classname": "CroppedEncoder",
            "encoder": encoder,
            "crop": crop,
        }

    def forward(self, x):
        x = crop(x, *self.crop)
        return self.encoder(x)