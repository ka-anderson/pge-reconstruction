import json
from os.path import join
import pathlib
import torch
import torchvision.transforms as T
from datasets.celebA import DatasetCelebA
from datasets.celebA_hq import DatasetCelebAHQ
from datasets.ffhq import DatasetFFHQ
import torch.nn as nn
from evaluation.eval_helpers import _torch_to_numpy_image, plot_image_grid

from misc_helpers.helpers import repo_dir
from datasets.noise_generator import GaussianNoiseGenerator
from evaluation.reconstruction import PGE_eval_from_folder, plot_PGE_images_from_model, plot_stats_from_folder_list
from models.model_loader import load_model_from_dict, load_model_from_folder
from models.style_gan import FinetunedStyleGenerator
from training.loss import DeepFaceLoss


EXP_GROUP = join("iclr_experiments", str(pathlib.Path(__file__).parent.resolve()).split("/")[-1])
BATCH_SIZE = 16

def eval_combined():
    G = FinetunedStyleGenerator(img_size=256)
    p_models = {
        "mse": load_model_from_folder(repo_dir("experiments", EXP_GROUP, "results", "auto_MSE"), "model_100", model_name="pre_gen").eval(),
        "fn": load_model_from_folder(repo_dir("experiments", EXP_GROUP, "results", "auto_FN"), "model_440", model_name="pre_gen").eval(),
    }

    PGE_eval_from_folder(
        model_path=repo_dir("experiments", EXP_GROUP, "results", "auto_FN"), 
        # dataset=DatasetCelebAHQ(batch_size=BATCH_SIZE, img_size=256, num_workers=2, pin_memory=True, to_zero_one=False),
        # dataset=DatasetCelebA(batch_size=BATCH_SIZE, img_size=256, num_workers=2, pin_memory=True, target_type="attr", to_zero_one=False),
        dataset=DatasetFFHQ(batch_size=BATCH_SIZE, img_size=256, num_workers=2, pin_memory=True, to_zero_one=False),
        p_models=p_models, 
        g_models={"": G},

        # noise_generator = GaussianNoiseGenerator((BATCH_SIZE,512)),
        # noise_seed=0,
        
        max_stat_batches=200,
        # plot_images=False,
        calc_stats=False,
        pretty_plots=True,
    )

def eval():
    G = FinetunedStyleGenerator(img_size=256)
    E = ArcfaceEncoder()

    model_path = repo_dir('experiments', EXP_GROUP, 'results', f'attr_MSE_fromnoise')
    # model_path = repo_dir('experiments', EXP_GROUP, 'output', f'arcF_FN+MSE_25000')
    p_models = {}
    for i in [170]:
        p_models[i] = load_model_from_folder(model_path, f"model_{i}", model_name="pre_gen").eval()

    PGE_eval_from_folder(
        model_path=model_path, 
        # dataset=DatasetCelebAHQ(batch_size=BATCH_SIZE, img_size=256, num_workers=2, pin_memory=True, to_zero_one=False),
        dataset=DatasetCelebA(batch_size=BATCH_SIZE, img_size=256, num_workers=2, pin_memory=True, target_type="attr", to_zero_one=False),
        # dataset=DatasetFFHQ(batch_size=BATCH_SIZE, img_size=256, num_workers=2, pin_memory=True, to_zero_one=False),
        p_models=p_models, 
        g_models={"": G},
        # e_models={"": E},

        # noise_generator=None, 
        # noise_generator=GaussianNoiseGenerator((BATCH_SIZE,512)),
        # noise_seed=0,
        stats_for_noise_input=False,
        
        max_stat_batches=1000,
        # plot_images=False,
        calc_stats=False,
        pretty_plots=True,
    )

class ArcfaceEncoder(nn.Module):
    def __init__(self):
        super(ArcfaceEncoder, self).__init__()
        self.deepface = DeepFaceLoss("ArcFace")

    def forward(self, image):
        return self.deepface.represent(image)




def plot():
    plot_stats_from_folder_list(
        folders=[repo_dir('experiments', EXP_GROUP, 'output', 'enc_MSE', 'eval_ffhq')],
        plot_folder=repo_dir('experiments', EXP_GROUP, 'output', 'enc_MSE', 'eval_ffhq'),
        x="model_name",
        metrics_to_plot=["i_MSE", "i_SSIM"],
    )
    

if __name__ == "__main__":
    eval()
    # eval_combined()
    # plot()
