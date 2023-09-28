import pathlib
import torch
from torch import nn
from torchvision.models import mobilenet_v2, resnet18
from tqdm import tqdm

from datasets.celebA import DatasetCelebA
from datasets.celebA_hq import DatasetCelebAHQ
from datasets.dataset_base import CustomDataset
from datasets.dataset_from_files import ImageFeatureDS
from datasets.noise_generator import GaussianNoiseGenerator
from misc_helpers.helpers import repo_dir
from models.model_loader import load_model_from_folder
from models.simple_models import CustomMobilenetV2, CustomResNet18, LeNet5
from models.style_gan import FinetunedStyleGenerator
from training.loss import DeepFaceLoss
from training.trainer import run_training
from training.train_interfaces.simple_training import ClassificationTrainInterface

from os.path import join
EXP_GROUP = join("iclr_experiments", str(pathlib.Path(__file__).parent.resolve()).split("/")[-1])



def classify():
    # dataset = DatasetCelebAHQ(batch_size=128, img_size=128, num_workers=2, pin_memory=True)

    dataset = DatasetCelebA(batch_size=128, img_size=256, num_workers=2, pin_memory=True, target_type=["bbox", "attr"], to_zero_one=False)

    # model = LeNet5(in_dim=3, out_dim=40, input_img_size=128)
    # model = CustomMobilenetV2(out_dim=40)
    # model = CustomMobilenetV2(out_dim=10177)

    model = CustomResNet18(out_dim=dataset.output_channels())

    # model = load_model_from_folder(repo_dir("experiments", EXP_GROUP, "results", "mobilenet_128_minus"), weights_file_name="model_49", model_name="model")

    train_interface = ClassificationTrainInterface(
        optimizer=torch.optim.Adam(model.parameters()),
        loss=nn.MSELoss(),
        # loss=nn.CrossEntropyLoss(),
        model=model,
    )

    run_training(
        dataset=dataset,
        seed=0, 
        exp_group=EXP_GROUP,
        exp_id=f'attrBB', 
        epochs=300, 
        # start_from_epoch=50,
        print_freq=1000,
        save_model=True,
        save_model_freq=100,

        train_interface=train_interface,

        # debug_distributed=True,

        verbose_mode=True,
        # gpus=[0, 1, 2, 3],
    )

def create_dataset():
    torch.manual_seed(0)

    with torch.no_grad():
        dataset = DatasetCelebA(batch_size=1, img_size=256, num_workers=2, pin_memory=True, target_type="attr", to_zero_one=False)._get_train_dataset()
        deepface = DeepFaceLoss("ArcFace")
        # enc = load_model_from_folder(repo_dir("experiments", EXP_GROUP, "results", "attr"), weights_file_name="model_300", model_name="model").cuda()
        # G = FinetunedStyleGenerator(img_size=256).cuda()
        # noise_gen = GaussianNoiseGenerator((128, 512))

        feature_list = []
        for i in range(10):
            print(f"{i} out of 10")
            for j in tqdm(range(5000)):
                image = dataset.__getitem__(i*5000 + j)[0].unsqueeze(0).cuda()
                feature = deepface(image)

                feature_list.append(feature)
        
            torch.save(feature_list, repo_dir("experiments", EXP_GROUP, "output", f"arcface_{i}.pt"))

def get_dataset(batch_size, num_subsets):
    '''
    num_subsets: how many 5000-element feature sets to use
    '''
    ds = ImageFeatureDS(
        features=torch.load(repo_dir("experiments", EXP_GROUP, "output", f"arcface_{num_subsets - 1}.pt"), map_location="cpu"),
        image_dataset=DatasetCelebA(batch_size=batch_size, img_size=256, num_workers=2, pin_memory=True, target_type="attr", to_zero_one=False)._get_train_dataset(),
    )
    ds.options.update({
        "feature_path": repo_dir("experiments", EXP_GROUP, "output"),
        "num_subsets": num_subsets,
    })
    customDS = CustomDataset(
        ds_id=f"image_feature_ds_celeba_arcface_{num_subsets}",
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        train_ds=ds,
    )

    return customDS

if __name__ == "__main__":
    classify()
    # create_dataset()
    # get_dataset(64, 5)