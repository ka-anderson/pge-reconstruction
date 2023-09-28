from matplotlib import pyplot as plt
import torch
from os.path import join
import json
import pandas as pd
import itertools
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
import os
import seaborn as sns
import torchvision.transforms as T

from tqdm import tqdm
from datasets.dataset_base import CustomDataset
from datasets.noise_generator import NoiseGenerator

from evaluation.eval_helpers import _torch_to_numpy_image, plot_image_grid, read_pd_markdown
from models.model_loader import load_model_from_dict, load_model_from_folder
from training.loss import DeepFaceLoss, FIDWrapper, FaceNetLoss, SixDRepAngleLoss, SixDRepPoseLoss
from torchvision.transforms.functional import center_crop, crop


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


METRICS = {
    "n":  {
        # "MSE": torch.nn.MSELoss(),
    },
    "f":  {
        "MSE": torch.nn.MSELoss(),
    },
    "i":  {
        "MSE": torch.nn.MSELoss(),
        "SSIM": StructuralSimilarityIndexMeasure(), # 1: perfect match, 0: no similarity, -1: perfect anticorrelation
        "DeepFace_VGG-Face": DeepFaceLoss("VGG-Face"),
        "DeepFace_OpenFace": DeepFaceLoss("OpenFace"),

        # "DeepFace_DeepFace": DeepFaceLoss("DeepFace"),
        # "DeepFace_ArcFace": DeepFaceLoss("ArcFace"),
        # "FID": FIDWrapper(), # low value -> small distance
        # "FaceNet": FaceNetLoss(), 
        # "DeepFace_Facenet": DeepFaceLoss("Facenet"),
        # "MS_SSIM": MultiScaleStructuralSimilarityIndexMeasure(normalize=None, betas=(0.05077184, 0.32367048, 0.34010333, 0.26779879)), # removed one scale (b_5 = 0.1333), adjusted the rest with b_i = (1 + b_5) * b_i so that they sum up to 1 again
        # "6D_Pose": SixDRepPoseLoss(),
        # "6D_yaw_pitch_roll": SixDRepAngleLoss(),
    },
}

def plot_stats_from_parent_dir(parent_dir_path, x, values_to_plot):
    out = []
    for f in os.scandir(parent_dir_path):
        if f.is_dir() and os.path.isfile(join(parent_dir_path, f.name, "stats.md")):
            out.append(f.path)
    
    plot_stats_from_folder_list(out, parent_dir_path, x, values_to_plot)


def plot_stats_from_folder_list(folders, plot_folder, x, metrics_to_plot):
    '''
    expects each folder to contain a "stats.md", obtained with calc_PGE_stats_from_model and an "opt.json"
    '''
    metric_keys = [f"{group_name}_{key}" for group_name, group_dict in METRICS.items() for key in group_dict.keys()]
    for metric_name in metrics_to_plot:
        assert metric_name in metric_keys

    combined_df = pd.DataFrame()

    for folder in folders:
        folder_df = read_pd_markdown(join(folder, "stats.md"))
        folder_df = folder_df.reset_index()
        folder_df[metrics_to_plot] = folder_df[metrics_to_plot].apply(pd.to_numeric, errors='coerce')
        folder_df=folder_df.dropna(axis=0)


        folder_df["folder"] = folder.split("/")[-1]
        combined_df = pd.concat([folder_df, combined_df])

    combined_metric_df = pd.DataFrame()

    pd.options.mode.chained_assignment = None
    for metric_name in metrics_to_plot:
        metric_df = combined_df[[x, "folder", metric_name]]
        metric_df["value"] = metric_df[metric_name]
        metric_df["metric_name"] = metric_name
        combined_metric_df = pd.concat([combined_metric_df, metric_df])
    
    combined_metric_df = combined_metric_df.sort_values(by=["folder"], ascending=True)

    sns.set_style("darkgrid")
    sns.relplot(data=combined_metric_df, x=x, y="value", hue="folder", kind="line", col="metric_name", marker="o", facet_kws={'sharey': "none", 'legend_out': True})
    plt.savefig(join(plot_folder, f"stats.png"))
    print(join(plot_folder, f"stats.png"))

def calc_PGE_stats_from_model(p_models, g_models, e_models, dataset:CustomDataset, noise_generator: NoiseGenerator, out_path, noise_seed=0, out_file_name="stats", max_num_batches=5000, stats_for_noise_input=True):
    def update_metrics(rec, target, metrics, calc_noise=False):
        value_types = ["n", "f", "i"] if calc_noise else ["f", "i"]

        for value_type in value_types:
            if value_type not in metrics:
                metrics[value_type] = {}

            for metric_id, fn in METRICS[value_type].items():
                metric_dict_key = f"{value_type}_{metric_id}"
                if metric_dict_key not in metrics[value_type]:
                    metrics[value_type][metric_dict_key] = 0

                metrics[value_type][metric_dict_key] += fn(rec[value_type], target[value_type])
        return metrics

    
    def calc_image_input_stats(dataloader, p, g, e, metrics, num_batches):
        for batch, (real_image, _) in enumerate(tqdm(dataloader, total=num_batches)):
            real_image = real_image.cuda()

            with torch.no_grad():
                real_f_vector = e(real_image)

                rec_image = g(p(real_f_vector))
                rec_f_vector = e(rec_image)

                metrics = update_metrics(rec={"f": rec_f_vector, "i": rec_image}, target={"f": real_f_vector, "i": real_image}, metrics=metrics)

                if batch == num_batches - 1:
                    return metrics
        
        raise Exception(f"Dataset did not contain enough samples to evaluate {num_batches} batches")
    
    num_train_batches = min(max_num_batches, int(len(dataset.get_train_dataloader().dataset) / dataset.batch_size) - 1)
    num_test_batches = min(max_num_batches, int(len(dataset.get_test_dataloader().dataset) / dataset.batch_size) - 1)

    df = pd.DataFrame()

    for _, group in METRICS.items():
        for _, metric in group.items():
            metric.cuda()

    for (p_name, p_model), (g_name, g_model), (e_name, e_model) in itertools.product(p_models.items(), g_models.items(), e_models.items()):
        torch.manual_seed(noise_seed)
        p_model.cuda()
        g_model.cuda()
        e_model.cuda()

        training_img_input_metrics, test_img_input_metrics, noise_input_metrics = {}, {}, {}
        print("Calculating Image Input Stats")
        training_img_input_metrics = calc_image_input_stats(dataset.get_train_dataloader(), p_model, g_model, e_model, training_img_input_metrics, num_train_batches)
        test_img_input_metrics = calc_image_input_stats(dataset.get_test_dataloader(), p_model, g_model, e_model, test_img_input_metrics, num_test_batches)
        
        if stats_for_noise_input:
            print("Calculating Noise Input Stats")

            for i in tqdm(range(num_train_batches)):
                with torch.no_grad():
                    noise = noise_generator.get_noise()
                    noise_image = g_model(noise)
                    noise_f = e_model(noise_image)

                    rec_noise = p_model(noise_f)
                    rec_image = g_model(rec_noise)
                    rec_f = e_model(rec_image)

                    update_metrics(rec={"f": rec_f, "i": rec_image, "n": rec_noise}, target={"f": noise_f, "i": noise_image, "n": noise}, metrics=noise_input_metrics, calc_noise=True)

        model_name = name_triple_to_string(p_name, g_name, e_name)
        all_stats = {"training_img_input": training_img_input_metrics, "test_img_input": test_img_input_metrics, "noise_input": noise_input_metrics}
        if stats_for_noise_input:
            all_stats.update({"noise_input": noise_input_metrics})


        for data_k, data_dict in all_stats.items():
            out = {"model_name": model_name, "input_data": data_k}
            num_batches = num_test_batches if data_k == "test_img_input" else num_train_batches

            for value_k, value_dict in data_dict.items():
                out.update({k: v.cpu().numpy() / num_batches for k, v in value_dict.items()})
            
            df = df.append(out, ignore_index=True)

        df.to_markdown(join(out_path, f'{out_file_name}.md'))
    print(join(out_path, f'{out_file_name}.md'))


def plot_PGE_images_from_model(p_dict, g_dict, e_dict, dataset: CustomDataset, noise_generator:NoiseGenerator, out_path, noise_seed=0, stats_for_noise_input=True, adjust_to_img_range=False, pretty_plots=False):  
    '''
    * adjust_to_img_rage: find the minimum and maximum value und set vmin/vmax accordingly. If false, min/max are set to -1,1
    '''
    def get_min_max(images):
        if adjust_to_img_range:
            min_value = min([torch.min(x) for x in images.values()])
            max_value = max([torch.max(x) for x in images.values()])

            if min_value != 0 or max_value != 1:
                print(f"Adjusting image for range [{min_value}, {max_value}]")
        else:
            min_value, max_value = -1, 1
        
        return min_value, max_value

    def plot_image_input_batch(dataloader, dataset_name):
        images = {}
        org_image_batch = next(iter(dataloader))[0].cuda()
        num_images = min(9, org_image_batch.shape[0])
        for i in range(num_images):
            images[f"{i}_target"] = _torch_to_numpy_image(org_image_batch[i, :, :, :])

        with torch.no_grad():
            for (p_name, p_model), (g_name, g_model), (e_name, e_model) in itertools.product(p_dict.items(), g_dict.items(), e_dict.items()):
                # if g_name != e_name:
                #     continue

                p_model.cuda(), g_model.cuda(), e_model.cuda()
                fmap_batch = e_model(org_image_batch)
                noise_batch = p_model(fmap_batch)

                rec_image_batch = g_model(noise_batch)

                # rec_image_batch = crop(rec_image_batch, 85, 64, 128, 128)
                # rec_image_batch = crop(rec_image_batch, 58, 38, 170, 170)

                for i in range(num_images):
                    images[name_triple_to_string(p_name, g_name, e_name, i)] = _torch_to_numpy_image(rec_image_batch[i, :, :, :])


        min_value, max_value = get_min_max(images)
        plot_image_grid(images, output_path=join(out_path, f"out_imginput_{dataset_name}.png"), ncols=9, nrows=len(images) // 9, norm_min=min_value, norm_max=max_value, pretty_plots=pretty_plots)

    torch.manual_seed(noise_seed)

    print("Creating images input plots")
    plot_image_input_batch(dataset.get_train_dataloader(), "trainset")
    plot_image_input_batch(dataset.get_test_dataloader(shuffle=False), "testset")
    if hasattr(dataset, "get_valid_dataloader"):
        plot_image_input_batch(dataset.get_valid_dataloader(), "valid")

    if not stats_for_noise_input:
        return

    print("Creating noise input images")
    images = {}
    noise_batch = noise_generator.get_noise()

    with torch.no_grad():
        for g_name, g_model in g_dict.items():
            g_model.cuda()
            gen_img_batch = g_model(noise_batch)
            for i in range(9):
                images[name_triple_to_string("", g_name, "", f"target_{i}")] = _torch_to_numpy_image(gen_img_batch[i, :, :, :])

            for (p_name, p_model), (e_name, e_model) in itertools.product(p_dict.items(), e_dict.items()):
                p_model.cuda(), e_model.cuda()

                gen_img_batch = g_model(noise_batch)
                feature_batch = e_model(gen_img_batch)

                rec_noise = p_model(feature_batch)
                rec_image_batch = g_model(rec_noise)

                for i in range(9):
                    images[name_triple_to_string(p_name, g_name, e_name, i)] = _torch_to_numpy_image(rec_image_batch[i, :, :, :])

    min_value, max_value = get_min_max(images)
    plot_image_grid(images, output_path=join(out_path, "out_noiseinput.png"), ncols=9, nrows=len(p_dict) + 1, norm_min=min_value, norm_max=max_value)


def PGE_eval_from_folder(model_path, dataset, noise_generator=None, out_path=None, p_models=None, g_models=None, e_models=None, noise_seed=0, plot_images=True, calc_stats=True, max_stat_batches=5000, stats_for_noise_input=True, pretty_plots=False):
    '''
    Create stats and images for a PGE-model. If p, g or e is not provided, it is taken from the options file of the model folder (as logged in the description of the train_interface), does not use the interface if all three are given.

    * noise gen: batchsize needs to be the same as it was during training (we are only drawing the first 9 images of the batch)
    * max_stat_batches: upper limit for the number of batches used to evaluate img stats. 
    '''

    if e_models == None or g_models == None or p_models == None:
        with open(join(model_path, "opt.json")) as file:
            options = json.load(file)

        traininterface_class = options["train_interface"]["class_group"] if "class_group" in options["train_interface"] else options["train_interface"]["class_name"]
        assert traininterface_class in ["ReconstrInterfaceP", "ReconstrInterfaceG", "ReconstrInterfaceP_old", "ReconstrInterfacePImage", "ReconstructionInterfacePEnsemble"], f"Reconstruction can not read data from a training interface of type {traininterface_class}."
        
        torch.manual_seed(noise_seed)

        # Encoder
        if e_models == None:
            if traininterface_class in ["ReconstrInterfaceP_old", "ReconstrInterfaceP"]:
                e_models = {"": load_model_from_dict(options["dataset"]["image_encoder"]).eval()}
            else:
                e_models = {"": load_model_from_dict(options["train_interface"]["enc"]).eval()}

        # Generator
        if g_models == None:
            if traininterface_class in ["ReconstrInterfaceP_old", "ReconstrInterfaceP"]:
                g_models = {"": load_model_from_dict(options["dataset"]["generator"]).eval()}
            else:
                g_models = {"": load_model_from_dict(options["train_interface"]["generator"]).eval()}

        # Pre-Generator
        if p_models == None:
            if traininterface_class in ["ReconstrInterfaceP_old", "ReconstrInterfaceP"]:
                p_models = {"": load_model_from_folder(model_path, model_name="pre_gen").eval()}
            else:
                p_models = {"": load_model_from_dict(options["train_interface"]["pre_gen"]).eval()}

    if out_path == None:
        out_path = model_path

    if plot_images:
        plot_PGE_images_from_model(p_models, g_models, e_models, dataset, noise_generator, out_path=out_path, noise_seed=noise_seed, adjust_to_img_range=False, stats_for_noise_input=stats_for_noise_input, pretty_plots=pretty_plots)
    if calc_stats:
        calc_PGE_stats_from_model(p_models, g_models, e_models, dataset, noise_generator, out_path, noise_seed, out_file_name="stats", max_num_batches=max_stat_batches, stats_for_noise_input=stats_for_noise_input)

def name_triple_to_string(p_name, g_name, e_name, prefix=""):
    # remove empty "" strings and convert to string
    names = list(filter(None, [str(name) for name in [prefix, p_name, g_name, e_name]]))
    return "_".join(names)
