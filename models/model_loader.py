from collections import OrderedDict
import os
import re
import torch
import json
from os.path import join

from models.pro_gan import ProDiscriminator, ProGenerator
from models.style_gan import FinetunedStyleDiscriminator, FinetunedStyleGenerator
from models.simple_models import MLP, ConvStack, CustomMobilenetV2, CustomResNet18, LeNet5, MLPEnsemble, TransposedConvStack

MODELS = {
    "ProGenerator": ProGenerator,
    "ProDiscriminator": ProDiscriminator,
    "FinetunedStyleDiscriminator": FinetunedStyleDiscriminator,
    "FinetunedStyleGenerator": FinetunedStyleGenerator,
    
    "MLP": MLP,
    "MLPEnsemble": MLPEnsemble, 
    "ConvStack": ConvStack,
    "TransposedConvStack": TransposedConvStack,
    "LeNet5": LeNet5,
    "CustomMobilenetV2": CustomMobilenetV2,
    "CustomResNet18": CustomResNet18,
}

def without_ddp_prefix(state_dict):
    '''
    ddp appends "module." to every weight
    https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
    '''
    model_dict = OrderedDict()
    pattern = re.compile('module.') 
    for k,v in state_dict.items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v    
        else:
            model_dict = state_dict
    return model_dict

def load_model_from_folder(folder_path, weights_file_name=None, model_name="generator", load_weights_from_opt=False):
    '''
    * model_name: name of the model in the training interface (where the model opts where saved)
    * weights_file_name: .pth is appended in this method. filename can be omitted if 
        (a) there is only one weights file in the folder, or 
        (b) the model options include a "weights_from" key (load_weights_from_opt=True)
    Note that a given weights_file_name overrides the load_weights_from_opt option - weights are loaded from the weights_file_name if one is given.
    '''
    with open(join(folder_path, "opt.json")) as opt_file:
        opt = json.loads(opt_file.read())
    
    model_opt = opt["train_interface"][model_name]
    model_name = model_opt["classname"]
    del model_opt["classname"]

    assert model_name in MODELS, f"{model_name} was not found in MODELS and might not be a loadable model."
    assert load_weights_from_opt == False or  "weights_from" in model_opt, "Cannot load weights from the opt file, there is no 'weights from' key."

    if weights_file_name != None:
        # weights are the result of this experiment (therefore saved in the model folder), and the name of the weights file is given
        weights_path = join(weights_folder(folder_path), f"{weights_file_name}.pth")

    elif "weights_from" in model_opt and load_weights_from_opt == True:
        # the weights were loaded from a different location before the training_interface started training
        weights_path = model_opt["weights_from"]
    else:
        # weights are the result of this experiment (therefore saved in the model folder), but there is only one weights file
        weights_path = None
        weights_path_parent = weights_folder(folder_path)
        for path in os.listdir(weights_path_parent):
            if path.endswith(".pth"):
                if weights_path != None:
                    raise Exception(f"There is more than one weights file in {folder_path}/model_weights, but the correct filename was not given")
                else:
                    weights_path = join(weights_path_parent, path)

    assert weights_path != None, "To load a model from a folder with more than one weights file, the path to the .pth file must be either part of the options file (weights_from), or given as function parameter (weights_file_name)"        

    if "weights_from" in model_opt:
        del model_opt["weights_from"]

    if "version" in model_opt:
        del model_opt["version"]

    model = MODELS[model_name](**model_opt)
    model.load_state_dict(without_ddp_prefix(torch.load(weights_path, map_location=torch.device('cpu'))))
    model.options.update({
        "weights_from": weights_path
    })

    return model

def weights_folder(path):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{path} does not exist")
    if os.path.isdir(join(path, "model_weights")):
        return join(path, "model_weights")
    if os.path.isdir(join(path, "model_weights_local")):
        return join(path, "model_weights_local")
    
    raise FileNotFoundError(f"There is no model_weights or model_weights_local folder in {path}.")
      
def load_model_from_dict(options_dict, weights_from=None):
    model_name = options_dict["classname"]
    
    assert model_name in MODELS, f"{model_name} was not found in MODELS and might not be a loadable model."
    assert weights_from!= None or "weights_from" in options_dict, "When loading a model from a dict, the weights to load from must be part of the dict (weights_from) or given as a parameter."

    weights_path = weights_from if weights_from != None else options_dict["weights_from"]

    options_dict = options_dict.copy()
    del options_dict["classname"]
    if "weights_from" in options_dict:
        del options_dict["weights_from"]
    if "version" in options_dict:  
        del options_dict["version"]

    model = MODELS[model_name](**options_dict)
    model.load_state_dict(without_ddp_prefix(torch.load(weights_path)))
    model.options.update({
        "weights_from": weights_path
    })

    return model
