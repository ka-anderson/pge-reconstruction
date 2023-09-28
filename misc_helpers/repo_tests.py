from math import cos, sin
from PIL import Image, ImageDraw
import numpy as np
import torch
import torchvision.transforms as T
import torchvision
from os.path import join
import cv2
from torchvision.transforms.functional import crop, center_crop
from facenet_pytorch import InceptionResnetV1, MTCNN

from deepface import DeepFace
# fix for DNN not found from https://stackoverflow.com/questions/56333388/tensorflow-fail-to-find-dnn-implementation
import tensorflow as tf

from datasets.celebA import DatasetCelebA
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
from sixdrepnet import SixDRepNet
from sixdrepnet import utils as sixDutils

# arcface
# from insightface.utils import ensure_available
# from insightface.model_zoo import model_zoo
# import onnx, onnxruntime

from datasets.ffhq import DatasetFFHQ
from misc_helpers.helpers import repo_dir
from training.loss import FaceNetLoss, MTCNNLoss, SixDRepAngleLoss, SixDRepPoseLoss

def test_SixDRepNet2():
    angle_loss = SixDRepPoseLoss()
    # angle_loss = SixDRepAngleLoss()
    dataset = DatasetCelebA(batch_size=1, img_size=256, to_zero_one=False)._get_test_dataset()
    img_a = dataset.__getitem__(64)[0].unsqueeze(0)
    img_b = dataset.__getitem__(1)[0].unsqueeze(0)
    img_c = dataset.__getitem__(73)[0].unsqueeze(0)


    torchvision.utils.save_image([img_a[0], img_b[0], img_c[0]], repo_dir("misc_helpers", "test_angle.png"), normalize=True)


    with torch.no_grad():
        l = angle_loss(img_a, img_b)
        # y, p, r = angle_loss(img_a, img_b)

    print(l)


def test_SixDRepNet1():
    model = SixDRepNet()
    loss = SixDRepPoseLoss()

    # img = T.ToTensor()(Image.open(repo_dir("datasets", "downloads", "ffhq", "256", "0", "00076.png"))).unsqueeze(0)

    dataset = DatasetFFHQ(batch_size=1, img_size=256, to_zero_one=False).get_train_dataloader()
    img = next(iter(dataset))[0]
    img = (img + 1)/2

    pred = loss.model(img)
    euler = sixDutils.compute_euler_angles_from_rotation_matrices(pred)*180/np.pi
    pitch = euler[:, 0].cpu().detach().numpy()
    yaw = euler[:, 1].cpu().detach().numpy()
    roll = euler[:, 2].cpu().detach().numpy()

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    # img = img[0].permute((1, 2, 0))
    # print(img.shape)

    tdx, tdy = 128, 128

    # X-Axis pointing to right. drawn in red
    x1 = 100 * (cos(yaw) * cos(roll)) + tdx
    y1 = 100 * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = 100 * (-cos(yaw) * sin(roll)) + tdx
    y2 = 100 * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = 100 * (sin(yaw)) + tdx
    y3 = 100 * (-cos(yaw) * sin(pitch)) + tdy

    pil_img = T.ToPILImage()(img[0])
    draw = ImageDraw.Draw(pil_img)
    draw.line((tdx, tdy, x1, y1), fill=128, width=5)
    draw.line((tdx, tdy, x2, y2), fill=128, width=5)
    draw.line((tdx, tdy, x3, y3), fill=256, width=5)
    pil_img.save(repo_dir("misc_helpers", "test.png"))

    # img = img.numpy()
    # img = model.draw_axis(img, y, p, r)

    # torchvision.utils.save_image(torch.from_numpy(img), repo_dir("misc_helpers", "test.png"))

def test_mtcnn():
    batch_size = 10
    to_tensor = T.PILToTensor()

    mtcnn = MTCNNLoss()
    dataset = DatasetFFHQ(batch_size=batch_size, img_size=128, to_zero_one=False).get_train_dataloader()
    image_batch = next(iter(dataset))[0]
    image_batch = (image_batch + 1)/2
    image_batch *= 255

    boxes, points = mtcnn.face_crop(image_batch)
    
    pil_images = []
    for i in range(batch_size):
        pil_img = T.ToPILImage()(image_batch[i])
        draw = ImageDraw.Draw(pil_img)

        draw.rectangle(boxes[i][:4].tolist(), width=3)
        draw.point(points[i].tolist())
        pil_images.append(255-to_tensor(pil_img).type(torch.float))
        print(pil_images)
    # pil_img.save(repo_dir("misc_helpers", "test_box.png"))
    torchvision.utils.save_image(pil_images, repo_dir("misc_helpers", "test_box.png"), normalize=True)

def test_arcface():
    # does not work
    resize = T.Resize((112, 112), antialias=None)
    mse = torch.nn.MSELoss()

    model_dir = ensure_available("arcface", 'buffalo_l', root=repo_dir("models", "pretrained"))
    # model = onnx.load(join(model_dir, "w600k_r50.onnx"))
    arcFaceONNX = model_zoo.get_model(name=join(model_dir, "w600k_r50.onnx"), root=model_dir)

    dataset = DatasetFFHQ(batch_size=1, img_size=128, to_zero_one=False).get_train_dataloader()
    image_batch = next(iter(dataset))[0]
    image_batch = resize(image_batch).numpy()
    out_1 = torch.tensor(arcFaceONNX.forward(image_batch))

    image_batch = next(iter(dataset))[0]
    image_batch = resize(image_batch).numpy()
    out_2 = torch.tensor(arcFaceONNX.forward(image_batch))

    print(mse(out_1, out_1))
    print(mse(out_2, out_2))
    print(mse(out_1, out_2))

def test_deepface():
    torch.manual_seed(1)
    dataset = DatasetFFHQ(batch_size=1, img_size=128, to_zero_one=False).get_train_dataloader()
    image_batch_a = (torch.permute(next(iter(dataset))[0][0], (1, 2, 0)) + 1)/2 * 255
    image_batch_b = (torch.permute(next(iter(dataset))[0][0], (1, 2, 0)) + 1)/2 * 255

    image_batch_a = image_batch_a.numpy()
    image_batch_b = image_batch_b.numpy()

    # image_batch_a = cv2.cvtColor(image_batch_a, cv2.COLOR_RGB2BGR)
    # image_batch_b = cv2.cvtColor(image_batch_b, cv2.COLOR_RGB2BGR)

    image_batch_a = image_batch_a.astype(np.uint8)
    image_batch_b = image_batch_b.astype(np.uint8)

    out = DeepFace.verify(image_batch_a, image_batch_b, model_name="Facenet", detector_backend="retinaface")
    print(out)

    # results = {}
    # for model_name in ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "ArcFace"]: # DeepID, Dlib, SFace
    #     print(model_name)
    #     # results[model_name] = DeepFace.verify(repo_dir("datasets", "downloads", "ffhq", "128", "0", "00076.png"), repo_dir("datasets", "downloads", "ffhq", "128", "0", "00077.png"), model_name=model_name)

    #     results[model_name] = DeepFace.verify(image_batch_a, image_batch_b, model_name=model_name)

    # print(results)

def test_facenet():
    torch.manual_seed(2)
    dataset = DatasetFFHQ(batch_size=128, img_size=256, to_zero_one=False).get_train_dataloader()
    img_a = next(iter(dataset))[0]
    img_b = next(iter(dataset))[0]

    fn_old = FaceNetLossOld()
    fn_new = FaceNetLoss(centercrop=128, manual_crop_window=None, resize=False, norm_min=-1, norm_max=1)

    print(fn_old(img_b, img_a))
    print(fn_new(img_b, img_a))

class FaceNetLossOld(torch.nn.Module):
    def __init__(self) -> None:
        super(FaceNetLossOld, self).__init__()

        self.facenet_model = InceptionResnetV1(pretrained="vggface2")
        self.loss = torch.nn.MSELoss()

        self.options = {
            "classname": "FaceNetLossOld",
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


if __name__ == "__main__":
    test_SixDRepNet2()