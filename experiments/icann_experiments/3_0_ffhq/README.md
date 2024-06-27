# 3.0 Image based reconstruction: FFHQ

For all experiments in this folder, both the encoder and the recontruction pipeline (G and P) have been trained on FFHQ, so we are assuming that E was trained on a public dataset. The evaluation is done on both FFHQ and CelebA - with CelebA being the interesting one. All of them are clean runs of 03_3_rec2, but with no prelayer.

Folder names: (target type)_(Loss function)

FN stands for FaceNet loss, MSE for pixelwise difference. FN+MSE is a model that started from the regular MSE trained one, and was continued with a combination of FN and MSE, with FN multiplied by 1000 to move both into a similar range.

Unless stated otherwise, the models are trained with the additional 0.001-weighted noise loss term - this is usually not important for the MSE loss, but is crucial to keep the FN loss from "breaking" the generator (see "attr_FN_noNoiseLoss"). The FN experiments start from models that were trained for MSE, for the same target type.

