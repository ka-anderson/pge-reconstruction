import torch

class NoiseGenerator():
    def __init__(self, shape, to_cuda=True) -> None:
        self.shape = shape
        self.to_cuda = to_cuda
        if to_cuda:
            self.device = "cuda"

        self.options = {
            "shape": shape,
            "to_cuda": to_cuda,
        }

    def to(self, device):
        self.device = device
        self.to_cuda = True
        return self

    def set_batch_size(self, batch_size):
        if self.shape[0] != batch_size:
            shape_list = list(self.shape)
            shape_list[0] = batch_size
            self.shape = tuple(shape_list)

    def get_noise(self):
        raise NotImplementedError()
    
class UniformNoiseGenerator(NoiseGenerator):
    def __init__(self, shape, to_cuda=True) -> None:
        super().__init__(shape, to_cuda)
        self.options.update({
            "classname": "UniformNoiseGenerator",
        })

    def get_noise(self):
        if self.to_cuda:
            return torch.rand(self.shape, dtype=torch.float, device=self.device)
        else:
            return torch.rand(self.shape, dtype=torch.float)
        
class GaussianNoiseGenerator(NoiseGenerator):
    def __init__(self, shape, to_cuda=True) -> None:
        super().__init__(shape, to_cuda)
        self.options.update({
            "classname": "GaussianNoiseGenerator",
        })

    def get_noise(self):
        if self.to_cuda:
            return torch.randn(self.shape, dtype=torch.float, device=self.device)
        else:
            return torch.randn(self.shape, dtype=torch.float)
    

class StyleGANGaussNoisegenerator(NoiseGenerator):
    def __init__(self, img_size, latent_dim, layers, batch_size, to_cuda=True) -> None:
        super().__init__(img_size, to_cuda)

        self.img_size = img_size
        self.latent_dim = latent_dim
        self.layers = layers
        self.batch_size = batch_size

        self.options.update({
            "classname": "StyleGANGaussNoisegenerator",
            "img_size": img_size,
            "latent_dim": latent_dim,
            "layers": layers,
            "batch_size": batch_size,
        })

    def img_noise(self):
        out = torch.FloatTensor(self.batch_size, self.img_size, self.img_size, 1).uniform_(0., 1.).cuda()
        if self.to_cuda: out = out.cuda()
        return out

    def noise_list(self, layers=None):
        if layers == None: layers = self.layers
        if self.to_cuda:
            return [(torch.randn(self.batch_size, self.latent_dim, device="cuda"), layers)]
        else:
            return [(torch.randn(self.batch_size, self.latent_dim), layers)]
    
    def mixed_list(self):
        tt = int(torch.rand(()).numpy() * self.layers)
        return self.noise_list(tt) + self.noise_list(self.layers - tt)
    

