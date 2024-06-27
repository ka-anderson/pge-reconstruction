from os.path import join
import pathlib
from datasets.celebA import DatasetCelebA

from misc_helpers.helpers import repo_dir
from evaluation.reconstruction import PGE_eval_from_folder, plot_stats_from_folder_list
from models.model_loader import load_model_from_folder
from models.style_gan import FinetunedStyleGenerator


EXP_GROUP = join("iclr_experiments", str(pathlib.Path(__file__).parent.resolve()).split("/")[-1])
ENC_FOLDER_AUTO = repo_dir("experiments", "iclr_experiments", "0_0_ffhq_autoencoder", "results", "100")
ENC_FOLDER_ATTR = repo_dir("experiments", "iclr_experiments", "0_1_ffhq_encoder", "results")
BATCH_SIZE = 16

def eval():
    G = FinetunedStyleGenerator(img_size=256)

    model_path = repo_dir('experiments', EXP_GROUP, 'output', "auto_FNc")
    p_models = {}
    for i in [330]:
        p_models[i] = load_model_from_folder(model_path, f"model_{i}", model_name="pre_gen")

    PGE_eval_from_folder(
        model_path=model_path, 
        dataset=DatasetCelebA(batch_size=BATCH_SIZE, img_size=256, num_workers=2, pin_memory=True, target_type="attr", to_zero_one=False),
        noise_generator=None, 
        p_models=p_models, 
        g_models={"": G},

        stats_for_noise_input=False,
        max_stat_batches=200,
        # plot_images=False,
        calc_stats=False,
        pretty_plots=True,
    )

def plot():
    plot_stats_from_folder_list(
        folders=[repo_dir('experiments', EXP_GROUP, 'output', 'enc_MSE', 'eval_ffhq')],
        plot_folder=repo_dir('experiments', EXP_GROUP, 'output', 'enc_MSE', 'eval_ffhq'),
        x="model_name",
        metrics_to_plot=["i_MSE", "i_SSIM"],
    )
    

if __name__ == "__main__":
    eval()
    # plot()

