import pickle
from typing import Literal, Optional, Union
from facenet_pytorch import InceptionResnetV1, MTCNN
from misc_helpers.detect_face import detect_face
import numpy as np
from torch import device
import torchvision.transforms as T
import importlib
import os
import torch
import torch.nn as nn
from torchmetrics.functional import structural_similarity_index_measure
from torchvision.transforms.functional import crop, center_crop
from misc_helpers.helpers import repo_dir
from models.model_helpers import freeze_parameters
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image

class StructuralDissimilarityIndexMeasure(nn.Module):
    '''
    -1 -> SSIM is 1, the images are exaclty the same.
    '''
    def __init__(self, *args, **kwargs) -> None:
        super(StructuralDissimilarityIndexMeasure, self).__init__()

        self.options = {
            "classname": "StructuralDissimilarityIndexMeasure",
        }

    def forward(self, x, y):
        return - structural_similarity_index_measure(x, y)

class DiscriminatorSimilarity(nn.Module):
    '''
    Loss based on (a) the output of a discriminator and (b) the similarity to a target image and

    loss = l * a + (1-l) * b

    use_total_loss: disc_loss is the mean output of the generator. Otherwise, the output is the difference in output between real and recreated images.
    '''
    def __init__(self, disc, l, image_loss_fn, use_total_loss=True) -> None:
        super(DiscriminatorSimilarity, self).__init__()

        self.l = l
        self.use_total_loss = use_total_loss
        self.image_loss_fn = image_loss_fn

        self.options = {
            "classname": "DiscriminatorSimilarity",
            "disc": disc,
            "image_loss_fn": image_loss_fn,
            "l": l,
            "use_total_loss": use_total_loss,
        }

    def to(self, device):
        self.disc = self.disc.to(device)
    
    def forward(self, pred, target):
        disc_out = self.disc(pred)
        if self.use_total_loss:
            disc_loss = - torch.mean(disc_out) / 3 # /3 to get to a similar range
        else:
            disc_out_real = self.disc(target)
            disc_loss = torch.mean(torch.abs(disc_out - disc_out_real))
        sim_loss = self.image_loss_fn(pred, target)

        return self.l * disc_loss + (1 - self.l) * sim_loss
    
class EncoderDisciminatorSimilarity(nn.Module):
    """
    Same as DiscriminatorSimilarity, but the image similarity is based on an encoder (assumes whitebox access to the encoder)
    """
    def __init__(self, disc, enc, l, enc_loss_fn) -> None:
        super(EncoderDisciminatorSimilarity, self).__init__()

        self.disc = disc.cuda()
        self.enc = freeze_parameters(enc.cuda())
        self.l = l
        self.enc_loss_fn = enc_loss_fn

        self.options = {
            "classname": "EncoderDisciminatorSimilarity",
            "disc": disc,
            "enc": enc,
            "enc_loss_fn": enc_loss_fn,
            "l": l,
        }

    def forward(self, pred, target):
        disc_out_rec = self.disc(pred)
        disc_loss = - torch.mean(disc_out_rec) / 3

        enc_out_rec = self.enc(pred)
        enc_out_real = self.enc(target)
        enc_loss = self.enc_loss_fn(enc_out_rec, enc_out_real)

        return self.l * disc_loss + (1 - self.l) * enc_loss
    
class EncoderLoss(nn.Module):
    '''
    Difference between enc(pred) and enc(target)
    '''
    def __init__(self, enc, enc_loss_fn) -> None:
        super(EncoderLoss, self).__init__()

        self.enc = freeze_parameters(enc)
        self.enc_loss_fn = enc_loss_fn

        self.options = {
            "classname": "EncoderLoss",
            "enc": enc,
            "enc_loss_fn": enc_loss_fn,
        }

    def to(self, device):
        self.enc = self.enc.to(device)

    def forward(self, pred, target):
        enc_out_rec = self.enc(pred)
        enc_out_real = self.enc(target)

        return self.enc_loss_fn(enc_out_rec, enc_out_real)

class ReconstructionLoss(nn.Module):
    def setup(self, input_batch):
        raise NotImplementedError()
    
class FIDWrapper(ReconstructionLoss):
    def __init__(self) -> None:
        super(FIDWrapper, self).__init__()
        self.fid = FrechetInceptionDistance(feature=64)

    def cuda(self):
        self.fid = self.fid.cuda()
        return self

    def forward(self, pred, target):
        pred = pred.type(torch.uint8)
        target = target.type(torch.uint8)

        self.fid.update(pred, real=False)
        self.fid.update(target, real=True)
        out = self.fid.compute()
        self.fid.reset()
        return out
    
class MTCNNLoss(ReconstructionLoss):
    def __init__(self,) -> None:
        super(MTCNNLoss, self).__init__()

        self.mtcnn = MTCNN(select_largest=True, keep_all=False)
        self.loss = nn.MSELoss()

        self.options = {
            "classname": "MTCNNLoss",
        }
    
    def cuda(self):
        self.mtcnn = self.mtcnn.cuda()
        return self

    def face_crop(self, img):
        img = ((img - torch.min(img)) / (torch.max(img) - torch.min(img))) * 255

        # need to use detect_face instead of detect, since the other forces no_grad
        boxes, points = detect_face(img, self.mtcnn.min_face_size, self.mtcnn.pnet, self.mtcnn.rnet, self.mtcnn.onet, self.mtcnn.thresholds, self.mtcnn.factor,self.mtcnn.device)

        return boxes, points
    

    def forward(self, pred_img, target_img):
        pred_box, pred_points = self.face_crop(pred_img)
        target_box, target_points = self.face_crop(target_img)

        return self.loss(pred_box, target_box) + self.loss(pred_points, target_points)

class SixDRepPoseLoss(ReconstructionLoss):
    '''
    https://github.com/thohemp/6DRepNet/tree/master

    version 2: converting the images from (-1, 1) to (0, 1)
    '''
    def __init__(self) -> None:
        super(SixDRepPoseLoss, self).__init__()

        SixDRepNetPackageModel = importlib.import_module("sixdrepnet.model")
        SixDRepNetPackageLoss = importlib.import_module("sixdrepnet.loss")

        self.model = SixDRepNetPackageModel.SixDRepNet(backbone_name='RepVGG-B1g2',
                        backbone_file='',
                        deploy=True,
                        pretrained=False)
        self.model.eval()
        self.model.load_state_dict(torch.load(repo_dir("models", "pretrained", "6DRepNet_300W_LP_AFLW2000.pth")))

        self.loss = SixDRepNetPackageLoss.GeodesicLoss()

        self.options = {
            "classname": "SixDRepPoseLoss",
            "version": 2,
        }

    def cuda(self):
        self.model = self.model.cuda()
        self.loss = self.loss.cuda()
        return self

    def forward(self, pred_img, target_img):
        pred_img = (pred_img + 1)/2
        target_img = (target_img + 1)/2

        r_pred = self.model(pred_img)
        r_target = self.model(target_img)

        return self.loss(r_target, r_pred)

class DeepFaceLoss(ReconstructionLoss):
    '''
    Super hacky implementation of Deepface, bypassing the strange limitation to evaluate single image at a time, being forced to use very heavy and slow preprocessing.

    Note that the model is a tensorflow model. This module is only for evaluation or to prepare datasets and can NOT be used for training!
    [40, 40, 100, 80]

    SFace and Dlib do not work at for my current implementation
    '''
    def __init__(self, model_name: Literal["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace"], manual_crop_window=None) -> None:
        super(DeepFaceLoss, self).__init__()

        # only import when using deepface, huge repo that requires a lot of overhead stuff
        # self.deepface = DeepFace
        self.deepface = importlib.import_module('deepface.DeepFace')
        self.tf = importlib.import_module('tensorflow')

        # this is a tensorflow model!
        self.model = self.deepface.build_model(model_name)
        target_size = self.deepface.functions.find_target_size(model_name)[0]
        self.resize = T.Resize((target_size, target_size), antialias=None)

        if manual_crop_window != None:
            self.crop_window = manual_crop_window
        else:
            self.crop_window = [target_size//4, target_size//4, target_size, target_size]
        self.distance_metric = torch.nn.MSELoss()

        self.options = {
            "classname": "DeepFaceLoss",
            "model_name": model_name,
        }

    def forward(self, imageA, imageB=None):
        if imageB != None:
            emb_a = self.represent(imageA)
            emb_b = self.represent(imageB)
            return self.distance_metric(emb_a, emb_b)
        return self.represent(imageA)

    def represent(self, image):
        image = crop(image, *self.crop_window)
        image = self.resize(image)
        image = torch.permute(image, (0, 2, 3, 1))

        # still a TF model
        image = self.tf.convert_to_tensor(image.cpu())
        return torch.tensor(self.model.predict(image)).cuda()

    def torch_to_cv(self, img_batch):
        if len(img_batch.shape) == 4:
            img_batch = torch.permute(img_batch, (0, 2, 3, 1))
        else:
            img_batch = torch.permute(img_batch, (1, 2, 0))

        img_batch = (img_batch + 1)/ 2 * 255
        img_batch = img_batch.cpu().numpy()
        # img_batch = cv2.cvtColor(img_batch, cv2.COLOR_RGB2BGR)
        img_batch = img_batch.astype(np.uint8)

        return img_batch
    
    def compare(self, pred_img, target_img):
        
        pred_img = self.torch_to_cv(pred_img)
        target_img = self.torch_to_cv(target_img)

        result = self.stat_base.copy()
        for i in range(pred_img.shape[0]):
            for model in result.keys():
                result[model] += self.deepface.verify(pred_img[i], target_img[i], model_name=model, detector_backend="retinaface")["distance"]

        result = {key: value/pred_img.shape[0] for key, value in result.items()}
        return result 

class FaceNetLoss(ReconstructionLoss):
    '''
    manual_crop_window: list with [top,left,height,width] to be applied to the image before feeding it to FaceNet

    version 2: 
        * added resize option - resize the images to 160x160px before embedding them - since the network was trained for this size, this is where the FaceNet performs best.
        Note that the resizing is done BEFORE the crop
        * using the "sum" reduction / BS on MSE for values that can be more easily compared to the original FaceNet
        
    version 3:
        * can either define a manual crop window (top, left, height, width) or a center crop window (certain size that will be cut out of the center, independent of original image size)
    
    version 4:
        * option to provide norm min and max (default, earlier implementation is the min/max found in the images)
        * went back to the old mean reduction
        * train mode!
    '''

    def __init__(self, manual_crop_window=[40, 40, 100, 80], centercrop=None, resize=True, norm_min=None, norm_max=None) -> None:
        super(FaceNetLoss, self).__init__()

        assert manual_crop_window == None or centercrop == None
        assert (norm_max == None and norm_min == None) or (norm_max != None and norm_min != None)

        self.facenet_model = InceptionResnetV1(pretrained="vggface2")
        # self.facenet_model = InceptionResnetV1(pretrained="vggface2", num_classes=64, classify=True).cuda() # <- V2
        self.loss = nn.MSELoss()
        self.manual_crop_window = manual_crop_window
        self.centercrop = centercrop
        self.resize = resize
        self.r = T.Resize((160, 160), antialias=None)

        self.norm_min = norm_min
        self.norm_max = norm_max

        self.options = {
            "classname": "FaceNetLoss",
            # see pytorch crop: top, left, height, width. calculated using the mean return of mtcnn (width +10 because it became to small)
            "manual_crop_window": manual_crop_window,
            "centercrop": centercrop,
            "resize": resize,
            "norm_min": norm_min,
            "norm_max": norm_max,
            "version": 4,
        }

    def cuda(self):
        self.facenet_model = self.facenet_model.cuda()
        return self

    def setup(self, img_batch):
        def size_adjustement(min_v, max_v):
            diff = max(0, 80 - (max_v - min_v))
            diff = np.ceil(diff / 2)

            return int(min_v - diff), int(max_v + diff)

        if self.resize:
            img_batch = self.r(img_batch)
            
        img_batch = img_batch.permute(0, 2, 3, 1) # mtcnn expects images in shape (bs, h, w, ch) and (always) converts them to the torch default (bs, ch, h, w)
        img_batch = self._norm(img_batch)
        mtccn = MTCNN(select_largest=True, keep_all=False)
        boxes, _ = mtccn.detect(img_batch)

        # mtccn returns different shapes (depending on how many faces it found), and all of them are weird.
        boxes = np.array([box[0].astype(int) for box in boxes])
        [xmin, ymin, xmax, ymax] = [boxes[:, i].mean() for i in range(4)]
        xmin, xmax = size_adjustement(xmin, xmax)
        ymin, ymax = size_adjustement(ymin, ymax)

        self.manual_crop_window = [ymin, xmin, ymax - ymin, xmax - xmin]
        self.options["mtcnn_crop_window"] = self.manual_crop_window

        print(f"FaceNetLoss updated for cropwindow {self.manual_crop_window}.")

    
    def _norm(self, img):
        if self.norm_min != None:
            return ((img - self.norm_min) / (self.norm_max - self.norm_min)) * 255
        if self.norm_min == None:
            return ((img - torch.min(img)) / (torch.max(img) - torch.min(img))) * 255

    def forward(self, pred_img, target_img):
        pred_img = self._norm(pred_img)
        target_img = self._norm(target_img)

        if self.resize:
            pred_img = self.r(pred_img)
            target_img = self.r(target_img)

        if self.manual_crop_window != None:
            pred_img = crop(pred_img, *self.manual_crop_window)
            target_img = crop(target_img,  *self.manual_crop_window)
        elif self.centercrop != None:
            pred_img = center_crop(pred_img, self.centercrop)
            target_img = center_crop(target_img, self.centercrop)

        pred_embedding = self.facenet_model(pred_img)
        target_embedding = self.facenet_model(target_img).detach()

        return self.loss(pred_embedding, target_embedding)
    
class FaceNetLossCenter(torch.nn.Module): 
    '''
    Simplified version fo FaceNetLoss.
    Assumes the images to be in (-1, 1) and compares the embedding for a 128x128 center crop
    '''
    def __init__(self) -> None:
        super(FaceNetLossCenter, self).__init__()

        self.facenet_model = InceptionResnetV1(pretrained="vggface2")
        self.loss = torch.nn.MSELoss()

        self.options = {
            "classname": "FaceNetLossCenter",
        }

    def to(self, device):
        self.facenet_model.to(device)
        return self

    def forward(self, pred_img, target_img):
        pred_img = ((pred_img + 1)/2) * 255
        target_img = ((target_img + 1)/2) * 255

        pred_img = center_crop(pred_img, 128)
        target_img = center_crop(target_img, 128)

        pred_embedding = self.facenet_model(pred_img)
        target_embedding = self.facenet_model(target_img).detach()

        return self.loss(pred_embedding, target_embedding)
    
class MSECrop(nn.Module):
    '''
    Crops only the prediction image (the first argument)
    '''
    def __init__(self, top, left, height, width, target_img_size) -> None:
        super(MSECrop, self).__init__()
        self.mse = torch.nn.MSELoss()
        self.crop_size = [top, left, height, width]
        self.resize = T.Resize((target_img_size, target_img_size), antialias=None)

        self.options = {
            "classname": "MSECrop",
            "crop_size": self.crop_size,
            "target_img_size": target_img_size,
        }

    def forward(self, pred_img, target_img):
        pred_img = crop(pred_img, *self.crop_size)
        pred_img = self.resize(pred_img)
        return self.mse(pred_img, target_img)


class FNplusMSE(torch.nn.Module):
    '''
    returns a tuple (mse + fn_mult * fn, mse, fn).
    
    ReconstrInterfaceImage can handle the tuple, other interfaces might (currently) not
    '''
    def __init__(self, fn_multiplier=1) -> None:
        super(FNplusMSE, self).__init__()

        self.mse = torch.nn.MSELoss()
        self.fn = FaceNetLoss()
        self.fn_multiplier = fn_multiplier

        self.options = {
            "classname":"FNplusMSE",
            "fn_multiplier": fn_multiplier,
            "fn": self.fn,
        }

    def to(self, device):
        self.fn.to(device)
        return self

    def forward(self, pred, target):
        mse = self.mse(pred, target)
        fn = self.fn(pred, target)

        return mse + self.fn_multiplier * fn, mse, fn


class GaussianLoss(nn.Module):
    def __init__(self, mean=0, std=1) -> None:
        super(GaussianLoss, self).__init__()

        self.mean = mean
        self.std = std

        self.options = {
            "classname": "GaussianLoss",
            "mean": mean,
            "std": std,
        }

    def forward(self, n):
        return torch.abs(torch.mean(n) - self.mean) + torch.abs(torch.std(n) - self.std)


class StyleDiscriminatorLoss(nn.Module):
    def __init__(self, pretrained_model_name="stylegan_ffhq_256"):
        super(StyleDiscriminatorLoss, self).__init__()
        with open(repo_dir("models", "pretrained", f"{pretrained_model_name}.pkl"), 'rb') as f:
            self.disc = pickle.load(f)["D"].cuda()
        self.resize = T.Resize((256, 256), antialias=None)
        self.options = {
            "classname": "DiscriminatorLoss",
            "pretrained_model_name": pretrained_model_name,
        }

    def forward(self, img, _):
        img = self.resize(img)
        return - torch.mean(self.disc(img, None))
