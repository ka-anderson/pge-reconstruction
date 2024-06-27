import json
import pathlib
import torch
from os.path import join
from datasets.ffhq import DatasetFFHQ


from models.model_loader import load_model_from_dict, load_model_from_folder
from models.simple_models import MLP
from models.style_gan import FinetunedStyleGenerator
from training.loss import FaceNetLoss, FaceNetLossCenter, GaussianLoss
from training.train_interfaces.reconstruction import ReconstrInterfaceImage
from training.trainer import run_training
from misc_helpers.helpers import repo_dir

EXP_GROUP = join("iclr_experiments", str(pathlib.Path(__file__).parent.resolve()).split("/")[-1])
ENC_FOLDER_AUTO = repo_dir("experiments", "iclr_experiments", "0_0_ffhq_autoencoder", "results", "100")
ENC_FOLDER_ATTR = repo_dir("experiments", "iclr_experiments", "0_1_ffhq_encoder", "results", "resnet")

BATCH_SIZE = 64

def train():
    G = FinetunedStyleGenerator(img_size=256)

    # pre_gen = MLP(out_channels=512, in_channels=32, hidden_channels=[512 for _ in range(2)], output_vector=True, activation="lrelu")
    pre_gen = load_model_from_folder(repo_dir("experiments", EXP_GROUP, "output", "enc_MSE"), weights_file_name="model_600", model_name="pre_gen")

    train_interface = ReconstrInterfaceImage(
        optimizer=torch.optim.Adam(pre_gen.parameters(), lr=0.0002),
        generator=G,
        pre_gen=pre_gen,

        encoder=load_model_from_folder(ENC_FOLDER_ATTR, model_name="model", weights_file_name="model_300"),
        # loss_fn=torch.nn.MSELoss(),
        loss_fn=FaceNetLossCenter(),
        noise_loss_fn=GaussianLoss()
    )

    run_training(
        dataset=DatasetFFHQ(batch_size=BATCH_SIZE, img_size=256, num_workers=1, pin_memory=True, to_zero_one=False),
        seed=0, 
        exp_group=EXP_GROUP,
        exp_id='enc_FN+NoiseLoss2', 
        epochs=3000, 
        train_interface=train_interface,
        print_freq=100,
        save_model=True,
        save_model_freq=10,

        gpus=[0, 1, 2, 3],
        # verbose_mode=True,
    ) 



if __name__ == "__main__":
    train()