import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as T
from torch.autograd.variable import Variable
from torch.nn import functional as F

from models.model_helpers import freeze_parameters
from models.pro_gan import ProDiscriminator, ProGenerator
from .train_interface_base import TrainInterface
from datasets.noise_generator import NoiseGenerator
from training.logger import Logger

class GANTrainInterface(TrainInterface):
    def __init__(self, generator: nn.Module, discriminator: nn.Module, noise_generator: NoiseGenerator, g_optimizer: torch.optim.Optimizer, d_optimizer: torch.optim.Optimizer, loss_fn) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.noise_generator = noise_generator
        self.loss_fn = loss_fn

        self.options = {
            'classname': 'GANTrainInterface',
            'generator': generator,
            'discriminator': discriminator,
            'g_optimizer': g_optimizer,
            'd_optimizer': d_optimizer,
            'noise_generator': noise_generator,
            'loss_fn': loss_fn,
        }

    def call_for_all_models(self, func):
        self.generator = func(self.generator)
        self.discriminator = func(self.discriminator)

    def get_model_out(self, input):
        return self.generator(self.noise_generator.get_noise())

    def get_learning_models(self):
        return {"gen": self.generator, "disc": self.discriminator}

    def train(self, dataloader: DataLoader, logger: Logger, epoch):
        size = len(dataloader.dataset)
        batch_size = int(dataloader.batch_size or 0)

        self.generator.train()
        self.discriminator.train()
        
        assert batch_size > 0

        for batch, (real_img, _) in enumerate(dataloader):
            real_img = Variable(real_img.to(self.device))

            # train the generator
            self.g_optimizer.zero_grad()

            generated_img = self.generator(self.noise_generator.get_noise())
            disc_out = self.discriminator(generated_img)
            generator_loss = self.loss_fn(disc_out, torch.ones((batch_size, 1), device=self.device))
            generator_loss.backward()
            self.g_optimizer.step()

            # train the discriminator
            self.d_optimizer.zero_grad()

            disc_out_real = self.discriminator(real_img) 
            disc_loss_real = self.loss_fn(disc_out_real, torch.ones((batch_size, 1), device=self.device))
            disc_loss_real.backward()

            generated_img = self.generator(self.noise_generator.get_noise()).detach() # disc needs a freshly generated image. Figuring this out took about 10 years off my life
            disc_out_fake = self.discriminator(generated_img)
            disc_loss_fake = self.loss_fn(disc_out_fake, torch.zeros((batch_size, 1), device=self.device))
            disc_loss_fake.backward()

            self.d_optimizer.step()

            self.print_substep(logger, batch * len(real_img), size, {"gen_loss": generator_loss, "disc_loss_real": disc_loss_real, "disc_loss_fake": disc_loss_fake})
        
        return {"gen_loss": generator_loss.item(), "disc_loss_real": disc_loss_real.item(), "disc_loss_fake": disc_loss_fake.item()}


class WGANTraininterface(GANTrainInterface):
    '''
    from the Wasserstein GAN github (terrible coding style): https://github.com/martinarjovsky/WassersteinGAN/blob/master/main.py
    and from here: https://www.kaggle.com/code/rafat97/pytorch-wasserstein-gan-wgan/notebook
    '''

    def __init__(self, generator: nn.Module, discriminator: nn.Module, noise_generator: NoiseGenerator, g_optimizer: torch.optim.Optimizer, d_optimizer: torch.optim.Optimizer, 
                 critic_clamp_upper, critic_clamp_lower, num_critic_updates) -> None:
        super().__init__(generator, discriminator, noise_generator, g_optimizer, d_optimizer, loss_fn=None)

        self.critic_clamp_upper = critic_clamp_upper
        self.critic_clamp_lower = critic_clamp_lower
        self.num_critic_updates = num_critic_updates

        self.options["classname"] = "WGANTraininterface"
        self.options.update({
            "critic_clamp_upper": critic_clamp_upper,
            "critic_clamp_lower": critic_clamp_lower,
            "num_critic_updates": num_critic_updates,
        })

    def _gen_loss(self, disc_out):
        return -1. * torch.mean(disc_out)
    
    def _disc_loss(self, disc_out_real, disc_out_fake):
        '''
        goal: negative answers for fake images, positive for real ones
        '''
        loss_fake = torch.mean(disc_out_fake)
        loss_real = torch.mean(disc_out_real)
        return loss_fake - loss_real, loss_fake, -loss_real

    def train(self, dataloader: DataLoader, logger: Logger, epoch):
        size = len(dataloader.dataset)
        batch_size = int(dataloader.batch_size or 0)

        self.generator.train()
        self.discriminator.train()

        for batch, (real_img, _) in enumerate(dataloader):
            # -- Train D --
            for param in self.generator.parameters():
                param.requires_grad = False
            for p in self.discriminator.parameters():
                p.requires_grad = True

            # clamp parameters to a cube
            for p in self.discriminator.parameters():
                p.data.clamp_(self.critic_clamp_lower, self.critic_clamp_upper)

            self.d_optimizer.zero_grad()
            disc_out_real = self.discriminator(real_img.to(self.device))
            generated_img = self.generator(self.noise_generator.get_noise()).detach()
            disc_out_fake = self.discriminator(generated_img)
            
            # also save loss on real and fake images separately for evaluation purposes
            disc_loss, disc_loss_fake, disc_loss_real = self._disc_loss(disc_out_real, disc_out_fake)
            disc_loss.backward()

            self.d_optimizer.step()

            # -- Train G --
            if batch % self.num_critic_updates == 0:
                for param in self.discriminator.parameters():
                    param.requires_grad = False
                for p in self.generator.parameters():
                    p.requires_grad = True

                self.g_optimizer.zero_grad()

                generated_img = self.generator(self.noise_generator.get_noise())
                disc_out = self.discriminator(generated_img)

                generator_loss = self._gen_loss(disc_out)
                generator_loss.backward()

                self.g_optimizer.step()

            self.print_substep(logger, batch * len(real_img), size, {"gen_loss": generator_loss, "disc_loss_total": disc_loss, "disc_loss_fake": disc_loss_fake, "disc_loss_real": disc_loss_real})
        
        return {"gen_loss": generator_loss.item(), "disc_loss_total": disc_loss.item(), "disc_loss_fake": disc_loss_fake.item(), "disc_loss_real": disc_loss_real.item()}


class ProGANTrainInterface(GANTrainInterface):
    def __init__(self, generator: ProGenerator, discriminator: ProDiscriminator, noise_generator: NoiseGenerator, g_optimizer: torch.optim.Optimizer, d_optimizer: torch.optim.Optimizer,
                 epoch_schedule: "list[(int, int)]") -> None:
        '''
        * epoch_schedule: a list with tuples [..., (m_i, n_i), ...] indicating that step i runs for m_i epochs, and outputs images with size n_i
        '''
        super().__init__(generator, discriminator, noise_generator, g_optimizer, d_optimizer, loss_fn=None)

        self.epoch_schedule = epoch_schedule
        self.generator.train()
        self.discriminator.train()

        self.options.update({
            "epoch_schedule": epoch_schedule,
        })
    
    def to(self, device, distributed=False):
        # if distributed: find_unused_parameters True, since we have layers that are not trained in every iteration
        return super().to(device, distributed, find_unused_parameters=distributed)

    def get_model_out(self, input):
        return self.generator(self.noise_generator.get_noise(), self.last_alpha, self.last_step)

    def _gen_loss(self, disc_out):
        return -1. * torch.mean(disc_out)
    
    def _gradient_penalty(self, real_img, generated_img, alpha, step):
        BATCH_SIZE, C, H, W = real_img.shape
        beta = torch.rand((BATCH_SIZE, 1, 1, 1), device=self.device).repeat(1, C, H, W)
        interpolated_images = real_img * beta + generated_img.detach() * (1 - beta)
        interpolated_images.requires_grad_(True)

        # Calculate critic scores
        mixed_scores = self.discriminator(interpolated_images, alpha, step)

        # Take the gradient of the scores with respect to the images
        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        return gradient_penalty
    
    def _disc_loss(self, disc_out_real, disc_out_fake, gp):
        '''
        goal: negative answers for fake images, positive for real ones
        '''
        loss_fake = torch.mean(disc_out_fake)
        loss_real = torch.mean(disc_out_real)
        loss_total = loss_fake - loss_real + gp + (0.001 * torch.mean(disc_out_real ** 2))
        return loss_total, loss_fake, -loss_real

    def _get_step(self, epoch):
        counted_epochs = 0
        for step, (num_e_in_step, img_size) in enumerate(self.epoch_schedule):
            interval_start = counted_epochs
            interval_stop = counted_epochs + num_e_in_step

            if epoch >= interval_start and epoch < interval_stop:
                epoch_in_step = epoch - interval_start # the position of the epoch relative to the current step
                num_e_in_half_step = num_e_in_step // 2

                # only the conv result (no residual) in: 
                #      the first step
                #      the second half of every step
                if step == 0 or epoch_in_step > num_e_in_half_step:
                    alpha = 1
                else:
                    alpha = epoch_in_step / (num_e_in_step - num_e_in_half_step) # num - halfnum in case of odd num

                return step, img_size, alpha

            counted_epochs = interval_stop
        raise Exception("Number of epochs does not match the schedule")


    def train(self, dataloader: DataLoader, logger: Logger, epoch):
        size = len(dataloader.dataset)
        batch_size = int(dataloader.batch_size or 0)        
        assert batch_size > 0

        self.noise_generator.set_batch_size(batch_size)

        step, img_size, alpha = self._get_step(epoch)
        self.last_step, self.last_img_size, self.last_alpha = step, img_size, alpha # save this for get_model_out
        resize = T.Resize((img_size, img_size), antialias=None)

        for batch, (real_img, _) in enumerate(dataloader):
            # Train D
            self.d_optimizer.zero_grad()

            real_img = resize(real_img.to(self.device))
            generated_img = self.generator(self.noise_generator.get_noise(), alpha, step).detach()

            disc_out_real = self.discriminator(real_img, alpha, step)
            disc_out_fake = self.discriminator(generated_img, alpha, step)


            gp = self._gradient_penalty(real_img, generated_img, alpha, step)
            disc_loss, disc_loss_fake, disc_loss_real = self._disc_loss(disc_out_real, disc_out_fake, gp)
            disc_loss.backward()
            self.d_optimizer.step()

            # Train G
            self.g_optimizer.zero_grad()

            generated_img = self.generator(self.noise_generator.get_noise(), alpha, step)
            disc_out = self.discriminator(generated_img, alpha, step)

            generator_loss = self._gen_loss(disc_out)
            generator_loss.backward()

            self.g_optimizer.step()

            self.print_substep(logger, batch * len(real_img), size, {"gen_loss": generator_loss, "disc_loss_total": disc_loss, "disc_loss_fake": disc_loss_fake, "disc_loss_real": disc_loss_real})
        
        return {"gen_loss": generator_loss.item(), "disc_loss_total": disc_loss.item(), "disc_loss_fake": disc_loss_fake.item(), "disc_loss_real": disc_loss_real.item(), "alpha": alpha, "step": step, "img_size": img_size} 

class GANFinetuningInterface(ProGANTrainInterface):
    def __init__(self, generator, discriminator, noise_generator: NoiseGenerator, update_disc, g_optimizer: torch.optim.Optimizer, d_optimizer: torch.optim.Optimizer = None, use_gp=True) -> None:
        '''
        use_gp: calculate gradient penalty. Use gp = 0 otherwise
        '''
        super().__init__(generator, discriminator, noise_generator, g_optimizer, d_optimizer, None)
        del self.options["epoch_schedule"]

        self.use_gp = use_gp
        self.update_disc = update_disc

        self.options.update({
            "classname": "GANFinetuningInterface",
            "use_gp": use_gp,
            "update_disc": update_disc,
        })

    def get_learning_models(self):
        out = {"pre_gen": self.generator.finetuning_layer}
        if self.update_disc:
            out["disc"] = self.discriminator
        return out
    
    def _gradient_penalty(self, real_img, generated_img):
        BATCH_SIZE, C, H, W = real_img.shape
        beta = torch.rand((BATCH_SIZE, 1, 1, 1), device=self.device).repeat(1, C, H, W)
        interpolated_images = real_img * beta + generated_img.detach() * (1 - beta)
        interpolated_images.requires_grad_(True)

        # Calculate critic scores
        mixed_scores = self.discriminator(interpolated_images)

        # Take the gradient of the scores with respect to the images
        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            # create_graph=True,
            # retain_graph=True,
        )[0]
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
        return gradient_penalty
        

    def train(self, dataloader: DataLoader, logger: Logger, epoch):
        size = len(dataloader.dataset)
        batch_size = int(dataloader.batch_size or 0)        
        assert batch_size > 0

        self.noise_generator.set_batch_size(batch_size)

        for batch, (real_img, _) in enumerate(dataloader):
            real_img = real_img.to(self.device)
            # Train D
            if self.update_disc:
                self.d_optimizer.zero_grad()

                generated_img = self.generator(self.noise_generator.get_noise()).detach()
                disc_out_real = self.discriminator(real_img)
                disc_out_fake = self.discriminator(generated_img)

                gp = self._gradient_penalty(real_img, generated_img) if self.use_gp else 0

                disc_loss, disc_loss_fake, disc_loss_real = self._disc_loss(disc_out_real, disc_out_fake, gp)
                disc_loss.requires_grad = True

                disc_loss.backward()
                self.d_optimizer.step()

            # Train G
            self.g_optimizer.zero_grad()

            generated_img = self.generator(self.noise_generator.get_noise())
            disc_out = self.discriminator(generated_img)

            generator_loss = self._gen_loss(disc_out)
            generator_loss.backward()

            self.g_optimizer.step()

            out = {"gen_loss": generator_loss}
            if self.update_disc:
                out["disc_loss_total"] = disc_loss
                out["disc_loss_fake"] = disc_loss_fake
                out["disc_loss_real"] = disc_loss_real
            self.print_substep(logger, batch * len(real_img), size, out)
        
        del out["time"]
        return {k: v.item() for k, v in out.items()}
    
class StyleGANFreezeDFinetuningInterface(ProGANTrainInterface):
    def __init__(self, generator, generator_r, discriminator, noise_generator: NoiseGenerator, g_optimizer: torch.optim.Optimizer, d_optimizer: torch.optim.Optimizer, freeze_index, gen_r_decay=0.999) -> None:
        '''
        freeze_index: layers from the input layer up to (and including) this layer index remain frozen (requires grad = False)
        '''
        super().__init__(generator, discriminator, noise_generator, g_optimizer, d_optimizer, None)

        self.generator_r = generator_r
        self.gen_r_decay = gen_r_decay
        generator_r.train(False)
        self._accumulate_gen(0)

        num_disc_layers = len(list(discriminator.model.named_children()))
        assert freeze_index <= num_disc_layers, f"Unable to freeze {freeze_index} layers, the discriminator has only {num_disc_layers} layers."
        frozen_layers = [f"b{2**(i+2)}" for i in range(num_disc_layers - 1, num_disc_layers - freeze_index, -1)]

        for name, module in discriminator.model.named_children():
            if name in frozen_layers:
                module = freeze_parameters(module)
    
        self.options.update({
            "classname": "StyleGANFreezeDFinetuningInterface",
            "freeze_index": freeze_index,
            "gen_r_decay": gen_r_decay,
        })

    def call_for_all_models(self, func):
        super().call_for_all_models(func)
        self.generator_r = func(self.generator_r)

    def get_learning_models(self):
        return {"gen": self.generator, "disc": self.discriminator, "gen_r": self.generator_r}

    def _gradient_penalty(self, real_img, generated_img):
        '''
        Slightly different to the ProGAN implementation, most notably not using an interpolation between real and generated images, but only real images
        copied from the freezeD code: https://github.com/sangwoomo/FreezeD/blob/b1725b5b65acecdf103ab917055f571bf161bfdf/stylegan/finetune.py#L280
        '''
        real_img.requires_grad_(True)
        disc_out = self.discriminator(real_img)

        gradient = torch.autograd.grad(
            inputs=real_img,
            outputs=disc_out.sum(),
            create_graph=True,
        )[0]

        gradient_penalty = gradient.view(gradient.shape[0], -1)
        gradient_penalty = (gradient_penalty.norm(2, dim=1) ** 2).mean()
        gradient_penalty = 10 / 2 * gradient_penalty

        # real_img.requires_grad_(False)

        return gradient_penalty

    def _disc_loss(self, disc_out_real, disc_out_fake, gp):
        '''
        goal: negative answers for fake images, positive for real ones
        Slightly different to the ProGAN version (most notably using SoftPlus)
        copied from the freezeD code: https://github.com/sangwoomo/FreezeD/blob/b1725b5b65acecdf103ab917055f571bf161bfdf/stylegan/finetune.py#L280
        '''
        loss_fake = torch.mean(F.softplus(disc_out_fake))
        loss_real = torch.mean(F.softplus(-disc_out_real))
        loss_total = loss_fake + loss_real + gp
        return loss_total, loss_fake, loss_real
    
    def _gen_loss(self, disc_out):
        return torch.mean(F.softplus(-disc_out))


    def _accumulate_gen(self, decay):
        gen_r = dict(self.generator_r.model.named_parameters())
        gen = dict(self.generator.model.named_parameters())

        for k in gen_r.keys():
            gen_r[k].data = gen_r[k].data * decay + gen[k].data * (1 - decay)
            # gen_r[k].data.mul_(decay).add_(gen[k].data, 1 - decay)


    def train(self, dataloader: DataLoader, logger: Logger, epoch):
        size = len(dataloader.dataset)
        batch_size = int(dataloader.batch_size or 0)        
        assert batch_size > 0

        self.noise_generator.set_batch_size(batch_size)

        for batch, (real_img, _) in enumerate(dataloader):
            real_img = real_img.to(self.device)

            # Train D
            self.d_optimizer.zero_grad()

            generated_img = self.generator(self.noise_generator.get_noise()).detach()
            disc_out_real = self.discriminator(real_img)
            disc_out_fake = self.discriminator(generated_img)

            gp = self._gradient_penalty(real_img, generated_img)

            disc_loss, disc_loss_fake, disc_loss_real = self._disc_loss(disc_out_real, disc_out_fake, gp)
            disc_loss.backward()
            self.d_optimizer.step()

            # Train G
            self.g_optimizer.zero_grad()

            generated_img = self.generator(self.noise_generator.get_noise())
            disc_out = self.discriminator(generated_img)

            generator_loss = self._gen_loss(disc_out)
            generator_loss.backward()

            self.g_optimizer.step()
            self._accumulate_gen(self.gen_r_decay)

            out = {"gen_loss": generator_loss}
            out["disc_loss_total"] = disc_loss
            out["disc_loss_fake"] = disc_loss_fake
            out["disc_loss_real"] = disc_loss_real
            self.print_substep(logger, batch * len(real_img), size, out)
        
        return {k: v.item() for k, v in out.items()}