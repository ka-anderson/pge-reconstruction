import pathlib
import importlib
import torch
from os.path import join
from datasets.celebA import DatasetCelebA
from datasets.ffhq import DatasetFFHQ


from models.model_loader import load_model_from_dict, load_model_from_folder
from models.style_gan import FinetunedStyleGenerator
from training.loss import FNplusMSE, FaceNetLoss, FaceNetLossCenter, GaussianLoss, SixDRepPoseLoss
from training.train_interfaces.reconstruction import ReconstrInterfaceImage, ReconstrInterfaceImageFromNoise
from training.trainer import run_training
from misc_helpers.helpers import repo_dir

EXP_GROUP = join("iclr_experiments", str(pathlib.Path(__file__).parent.resolve()).split("/")[-1])
ENC_FOLDER_AUTO = repo_dir("experiments", "iclr_experiments", "0_2_celebA_autoencoder", "results", "100")
ENC_FOLDER_ATTR = repo_dir("experiments", "iclr_experiments", "0_3_celebA_encoder", "results", "attr") # 40
ENC_FOLDER_ATBB = repo_dir("experiments", "iclr_experiments", "0_3_celebA_encoder", "results", "attrBB") # 44
ENC_FOLDER_IDEN = repo_dir("experiments", "iclr_experiments", "0_3_celebA_encoder", "results", "iden_CE") # 10177

BATCH_SIZE = 64

def train_with_enc():
    G = FinetunedStyleGenerator(img_size=256)

    # pre_gen = MLP(out_channels=512, in_channels=10177, hidden_channels=[512 for _ in range(2)], output_vector=True, activation="lrelu")
    pre_gen = load_model_from_folder(repo_dir("experiments", EXP_GROUP, "output", "iden_MSE"), weights_file_name="model_30", model_name="pre_gen")

    train_interface = ReconstrInterfaceImage(
    # train_interface = ReconstrInterfaceImageFromNoise(
        optimizer=torch.optim.Adam(pre_gen.parameters(), lr=0.0002),
        generator=G,
        pre_gen=pre_gen,

        encoder=load_model_from_folder(ENC_FOLDER_IDEN, model_name="model", weights_file_name="model_200"),
        # encoder=load_model_from_folder(ENC_FOLDER_AUTO, model_name="encoder", weights_file_name="encoder_300"),
        # encoder=load_model_from_folder(ENC_FOLDER_ATTR, model_name="model", weights_file_name="model_100"),
        # encoder=load_model_from_folder(ENC_FOLDER_ATBB, model_name="model", weights_file_name="model_299"),

        loss_fn=torch.nn.MSELoss(),
        # loss_fn=FaceNetLoss(),
        # loss_fn=FaceNetLossCenter(),
        noise_loss_fn=GaussianLoss(),
    )

    run_training(
        dataset=DatasetFFHQ(batch_size=BATCH_SIZE, img_size=256, num_workers=1, pin_memory=True, to_zero_one=False),
        # dataset=None,
        seed=0, 
        exp_group=EXP_GROUP,
        exp_id='iden_MSE_continued', 
        epochs=3000, 
        train_interface=train_interface,
        print_freq=300,
        save_model=True,
        save_model_freq=10,

        # gpus=[0, 1, 2, 3],
        verbose_mode=True,
    ) 



def train_with_ds():
    G = FinetunedStyleGenerator(img_size=256)
    celeb_enc = importlib.import_module('experiments.iclr_experiments.0_3_celebA_encoder.train_0_3',)
    dataset = celeb_enc.get_dataset(BATCH_SIZE, 5)


    # pre_gen = MLP(out_channels=512, in_channels=512, hidden_channels=[512 for _ in range(2)], output_vector=True, activation="lrelu")
    pre_gen = load_model_from_folder(repo_dir("experiments", EXP_GROUP, "output", "arcF_MSE_25000"), weights_file_name="model_300", model_name="pre_gen")

    train_interface = ReconstrInterfaceImage(
        optimizer=torch.optim.Adam(pre_gen.parameters(), lr=0.0002),
        generator=G,
        pre_gen=pre_gen,

        # loss_fn=torch.nn.MSELoss(),
        # loss_fn=FaceNetLoss(),
        loss_fn=FNplusMSE(),

    )

    run_training(
        dataset=dataset,
        seed=0, 
        exp_group=EXP_GROUP,
        exp_id='arcF_FN+MSE_25000', 
        epochs=1000, 
        train_interface=train_interface,
        print_freq=300,
        save_model=True,
        save_model_freq=10,

        # debug_distributed=True,
        # gpus=[0, 1, 2, 3],
        verbose_mode=True,
    ) 



if __name__ == "__main__":
    train_with_enc()