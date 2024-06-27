from os.path import join
import pathlib
import torch
from datasets.celebA import DatasetCelebA

from misc_helpers.helpers import repo_dir
from evaluation.reconstruction import PGE_eval_from_folder
from models.model_helpers import IdentityModule
from models.model_loader import load_model_from_folder
from models.simple_models import MLPEnsemble
from models.style_gan import FinetunedStyleGenerator

import importlib
Inversion = importlib.import_module("baselines.1_background_know.model").Inversion

EXP_GROUP = join("iclr_experiments", str(pathlib.Path(__file__).parent.resolve()).split("/")[-1])
ENC_FOLDER_AUTO = repo_dir("experiments", "iclr_experiments", "0_2_celebA_autoencoder", "results", "100")
ENC_FOLDER_ATTR = repo_dir("experiments", "iclr_experiments", "0_3_celebA_encoder", "results", "attr") # 40
ENC_FOLDER_IDEN = repo_dir("experiments", "iclr_experiments", "0_3_celebA_encoder", "results", "iden_CE") # 10177
BATCH_SIZE = 16

def combined_P(path_0, model_name_0, path_1, model_name_1, path_2, model_name_2):
    P = MLPEnsemble(
        num_mlps=3, out_channels=512, in_channels=40, activation="lrelu", final_activation="none", hidden_channels=[512 for _ in range(2)],
        # output_mapping=[1,1,0,0,0,0,1,1,1,1,1,1,1,1],
        output_mapping=[0,0,1,1,1,1,2,2,2,2,2,2,2,2],
    )
    P.components[0] = load_model_from_folder(path_0, model_name_0, model_name="pre_gen")
    P.components[1] = load_model_from_folder(path_1, model_name_1, model_name="pre_gen")
    P.components[2] = load_model_from_folder(path_2, model_name_2, model_name="pre_gen")

    return P

def combine():
    
    G = EnsembleGenerator(in_channels=512, img_size=256)

    # attr
    celeb_dir = repo_dir('experiments', "iclr_experiments", "3_1_celebA", "results")
    path_init, model_name_init = join(celeb_dir, f'attr_MSE_fromnoise'), "model_-1"
    path_mse, model_name_mse = join(celeb_dir, f'attr_MSE'), "model_1770"
    path_fn, model_name_fn = join(celeb_dir, f'attr_FN'), "model_440"
    E = load_model_from_folder(ENC_FOLDER_ATTR, model_name="model", weights_file_name="model_300")

    P_INIT = combined_P(path_init, model_name_init, path_init, model_name_init, path_init, model_name_init)
    P_MSE = combined_P(path_mse, model_name_mse, path_mse, model_name_mse, path_mse, model_name_mse)
    P_FN = combined_P(path_fn, model_name_fn, path_fn, model_name_fn, path_fn, model_name_fn)
    P_MSE_FN = combined_P(path_mse, model_name_mse, path_fn, model_name_fn, path_mse, model_name_mse)
    P_YANG = Inversion(nc=3, ngf=128, nz=40, truncation=40, c=50.).cuda()
    P_YANG.load_state_dict(torch.load(repo_dir("baselines", "1_background_know", "output", "celeba_ffhq", "model_weights_local", f'inversion_model_2.pth')))

    PGE_eval_from_folder(
        model_path=repo_dir("experiments", EXP_GROUP), 
        # dataset=DatasetFFHQ(batch_size=BATCH_SIZE, img_size=256, to_zero_one=False),
        dataset=DatasetCelebA(batch_size=BATCH_SIZE, img_size=256, target_type="attr", to_zero_one=False),
        # noise_generator=GaussianNoiseGenerator((BATCH_SIZE, 512)), 
        p_models={"init": P_INIT, "mse": P_MSE, "fn": P_FN, "mse+fn": P_MSE_FN, "yang": P_YANG}, 
        g_models={"ours": G, "yang": IdentityModule()},
        e_models={"": E},

        stats_for_noise_input=False,
        max_stat_batches=5000, 
        # plot_images=False, 
        calc_stats=False,
        # deepface_stats=True,
        pretty_plots=True,
    )


class EnsembleGenerator(FinetunedStyleGenerator):
    def __init__(self, in_channels=512, pretrained_in_channels=512, img_size=64, pretrained_model_path="stylegan_ffhq_256", pretrained_img_size=256, styleGAN_seed=0, fuzzy_lambda=0):
        super().__init__(in_channels, pretrained_in_channels, img_size, pretrained_model_path, pretrained_img_size, styleGAN_seed)
        self.fuzzy_lambda = fuzzy_lambda

    def fuzzy(self, latent):
        if self.fuzzy_lambda == 0:
            return latent.repeat_interleave(self.styles_per_mlp, 1)

        latent_fuzzy = (1 - self.fuzzy_lambda) * latent + self.fuzzy_lambda * torch.mean(latent, dim=1, keepdim=True)
        return latent_fuzzy.repeat_interleave(self.styles_per_mlp, 1)

    def forward(self, n_list):

        # styles_per_mlp = 14 // len(n_list)
        latent_list = [self.model.mapping(n, None)[:, 0, :] for n in n_list]        
        latent = torch.stack(latent_list, dim=1)
        if self.fuzzy_lambda != 0:
            latent = (1 - self.fuzzy_lambda * latent) + self.fuzzy_lambda * torch.mean(latent, dim=1, keepdim=True)

        pred_images = self.model.synthesis(latent)

        return pred_images

    
if __name__ == "__main__":
    # single_exp()
    combine()
    # eval_angles()



