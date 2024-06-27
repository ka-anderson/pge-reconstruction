from os.path import join
import pathlib
import torch
from datasets.celebA import DatasetCelebA

from misc_helpers.helpers import repo_dir
from datasets.noise_generator import GaussianNoiseGenerator
from evaluation.reconstruction import PGE_eval_from_folder
from models.model_loader import load_model_from_folder
from models.simple_models import MLPEnsemble
from models.style_gan import FinetunedStyleGenerator
from training.loss import MTCNNLoss

EXP_GROUP = join("iclr_experiments", str(pathlib.Path(__file__).parent.resolve()).split("/")[-1])
ENC_FOLDER_AUTO = repo_dir("experiments", "iclr_experiments", "0_0_ffhq_autoencoder", "results", "100")
ENC_FOLDER_ATTR = repo_dir("experiments", "iclr_experiments", "0_1_ffhq_encoder", "results", "resnet")
BATCH_SIZE = 16

def combine():
    def combined_P(path_1, model_name_1, path_2, model_name_2):
        P = MLPEnsemble(
            num_mlps=2, out_channels=512, in_channels=32, activation="lrelu", final_activation="none", hidden_channels=[512 for _ in range(2)],
            # output_mapping=[1,1,0,0,0,0,1,1,1,1,1,1,1,1],
            output_mapping=[0,0,1,1,1,1,0,0,0,0,0,0,0,0],
        )
        P.components[0] = load_model_from_folder(path_1, model_name_1, model_name="pre_gen")
        P.components[1] = load_model_from_folder(path_2, model_name_2, model_name="pre_gen")
    
        return P


    G = EnsembleGenerator(in_channels=512, img_size=256)
    G.fuzzy_lambda = 0
    E = load_model_from_folder(ENC_FOLDER_ATTR, model_name="model", weights_file_name="model_300")

    base_path = repo_dir('experiments', "iclr_experiments", "3_0_ffhq", 'results')
    path_mse, model_name_mse = join(base_path, 'attr_MSE'), "model_1710"
    path_fn, model_name_fn = join(base_path, 'attr_FN'), "model_800"

    P_MSE = combined_P(path_mse, model_name_mse, path_mse, model_name_mse)
    P_FN = combined_P(path_fn, model_name_fn, path_fn, model_name_fn)
    P_MIX = combined_P(path_mse, model_name_mse, path_fn, model_name_fn)

    PGE_eval_from_folder(
        model_path=repo_dir("experiments", EXP_GROUP), 
        # dataset=DatasetFFHQ(batch_size=BATCH_SIZE, img_size=256, to_zero_one=False),
        dataset=DatasetCelebA(batch_size=BATCH_SIZE, img_size=256, target_type="attr", to_zero_one=False),
        p_models={"mse": P_MSE, "fn": P_FN, "mix": P_MIX}, 
        g_models={"": G},
        e_models={"": E},

        stats_for_noise_input=False,
        max_stat_batches=5000, 
        plot_images=False, 
        # calc_stats=False,
        pretty_plots=True,
    )


class EnsembleGenerator(FinetunedStyleGenerator):
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
    combine()



