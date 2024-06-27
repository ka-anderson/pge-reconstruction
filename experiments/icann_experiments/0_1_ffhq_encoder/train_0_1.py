import pathlib
import torch
from torch import nn
from tqdm import tqdm

from datasets.ffhq import DatasetFFHQAttributes
from misc_helpers.helpers import repo_dir
from models.model_helpers import freeze_parameters
from models.model_loader import load_model_from_folder
from models.simple_models import CustomMobilenetV2, CustomResNet18, LeNet5
from models.style_gan import FinetunedStyleGenerator
from training.trainer import run_training
from training.train_interfaces.simple_training import ClassificationTrainInterface

from os.path import join

EXP_GROUP = join("iclr_experiments", str(pathlib.Path(__file__).parent.resolve()).split("/")[-1])

def classify():
    dataset = DatasetFFHQAttributes(batch_size=128, img_size=256, num_workers=2, pin_memory=True, to_zero_one=False, load_images=False)

    # model = LeNet5(in_dim=3, out_dim=40, input_img_size=128)
    # model = CustomMobilenetV2(out_dim=32)
    # model = CustomMobilenetV2(out_dim=10177)
    model = CustomResNet18(out_dim=32)

    train_interface = ClassificationTrainInterface(
        optimizer=torch.optim.Adam(model.parameters()),
        loss=nn.MSELoss(),
        model=model,
    )

    run_training(
        dataset=dataset,
        seed=0, 
        exp_group=EXP_GROUP,
        exp_id=f'resenet', 
        epochs=300, 
        print_freq=100,
        save_model=True,
        save_model_freq=25,

        train_interface=train_interface,
        gpus=[0, 1, 2, 3],
    )

def create_dataset():
    torch.manual_seed(0)

    with torch.no_grad():
        dataset = DatasetFFHQAttributes(batch_size=128, img_size=256, num_workers=2, pin_memory=True, to_zero_one=False, load_images=False)._get_train_dataset()
        enc = load_model_from_folder(repo_dir("experiments", EXP_GROUP, "results", "resnet"), weights_file_name="model_300", model_name="model").cuda()

        feature_list = []
        for i in tqdm(range(len(dataset))):
            image = dataset.__getitem__(i)[0]
            image = torch.unsqueeze(image, 0).cuda()
            feature = enc(image)
            feature_list.extend(feature)
        
        torch.save(feature_list, repo_dir("experiments", EXP_GROUP, "results", "attr_features_resnet.pt"))



if __name__ == "__main__":
    # classify()
    create_dataset()