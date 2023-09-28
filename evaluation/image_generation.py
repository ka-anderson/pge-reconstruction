import os
import pandas as pd
import torch
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm
from os.path import join

from datasets.dataset_base import CustomDataset
from datasets.noise_generator import NoiseGenerator
from evaluation.eval_helpers import _torch_to_numpy_image, plot_image_grid
from models.model_helpers import FrechetModule

def calc_generator_stats(model, frechet_model: FrechetModule, dataset: CustomDataset, noise_gen: NoiseGenerator, output_path=None, save_to_file=True):
    def calc_fid(dataloader, use_same_img=False):
        if use_same_img:
            last_real_img_batch = None
        fid = FrechetInceptionDistance(feature=frechet_model, reset_real_features=False).cuda()

        for _, (real_img, _) in enumerate(tqdm(dataloader)):
            real_img = real_img.cuda()
            if use_same_img:
                if last_real_img_batch == None:
                    last_real_img_batch = real_img
                    continue
                else:
                    generated_img = last_real_img_batch
                    last_real_img_batch = real_img
            else:
                noise = noise_gen.get_noise()
                generated_img = model(noise)

            fid.update(real_img, real=True)
            fid.update(generated_img, real=False)

        return fid.compute().item()

    def calc_ssim(dataloader, use_same_img=False):
        if use_same_img:
            last_real_img_batch = None
        ssim = StructuralSimilarityIndexMeasure().cuda()
        ssim_sum = 0
        for _, (real_img, _) in enumerate(tqdm(dataloader)):
            real_img = real_img.cuda()
            if use_same_img:
                if last_real_img_batch == None:
                    last_real_img_batch = real_img
                    continue
                else:
                    generated_img = last_real_img_batch
                    last_real_img_batch = real_img
            else:
                noise = noise_gen.get_noise()
                generated_img = model(noise)

            ssim_sum += ssim(generated_img, real_img).item()

        return ssim_sum / len(dataloader.dataset)

    torch.manual_seed(0)
    frechet_model.set_as_fid_model()
    frechet_model.to("cuda")
    model.to("cuda")

    dataloader_train = dataset.get_train_dataloader()
    dataloader_test = dataset.get_test_dataloader()

    with torch.no_grad():
        print("Evaluating FID")
        fid_results = {
            # "base": calc_fid(dataloader_train, use_same_img=True),
            "train": calc_fid(dataloader_train),
            "test": calc_fid(dataloader_test),
        }

        print("Evaluating SSIM")
        ssim_results = {
            # "base": calc_ssim(dataloader_train, use_same_img=True),
            "train": calc_ssim(dataloader_train),
            "test": calc_ssim(dataloader_test),
        }


    if save_to_file:
        df = pd.DataFrame({
            "fid": fid_results,
            "ssim": ssim_results,
        })
        df.to_markdown(join(output_path, 'stats.md'))
        print(join(output_path, 'stats.md'))

    return fid_results, ssim_results

def calc_generator_stats_multi(models, frechet_model: FrechetModule, dataset: CustomDataset, noise_gen: NoiseGenerator, output_path):
    model_dfs = []
    
    for name, model in models.items():
        fid_results, ssim_results = calc_generator_stats(model, frechet_model, dataset, noise_gen, output_path, save_to_file=False)

        model_df = pd.DataFrame({
            "fid_train": fid_results["train"],
            "fid_test": fid_results["test"],
            "ssim_train": ssim_results["train"],
            "ssim_test": ssim_results["test"],
        }, index=[str(name)])
        model_dfs.append(model_df)

        df = pd.concat(model_dfs)
        df.to_markdown(join(output_path, 'stats.md'))
        print(join(output_path, 'stats.md'))
    print("Done!")

def plot_generator_results(models, dataset: CustomDataset, output_path, noise_gen: NoiseGenerator, output_file_name="out", num_reconstructions: int=5):
    '''
    returns a row of num_reconstructions targets, and num_reconstructions*num_reconstructions example outputs
    '''

    images = {}
    target_image_batch = next(iter(dataset.get_train_dataloader()))[0]

    for i in range(num_reconstructions):
        images[f"{i}_target"] = _torch_to_numpy_image(target_image_batch[i, :, :, :])

    if type(models) == dict:
        for key, model in tqdm(models.items()):
            generated_image_batch = model(noise_gen.get_noise())
            for i in range(num_reconstructions):
                images[f"{key}_{i}"] = _torch_to_numpy_image(generated_image_batch[i, :, :, :])

        num_rows = len(models) + 1

    else: # one single model
        generated_image_batch = models(noise_gen.get_noise())
        for i in range(num_reconstructions**2):
            images[f"rec_{i}"] = _torch_to_numpy_image(generated_image_batch[i, :, :, :])

        num_rows = num_reconstructions ** 2 + 1


    plot_image_grid(images, output_path=join(output_path, f"{output_file_name}.png"), ncols=num_reconstructions, nrows=num_rows, norm_min=torch.min(target_image_batch), norm_max=torch.max(target_image_batch))