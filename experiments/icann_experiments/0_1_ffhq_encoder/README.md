# Target: FFHQ attribute encoder

The encoder is trained on the same data as the generator, making this another baseline. It can be used to find the difference between an autoencoder and an attribute encoder, given that they are able to create the exact kind of image that is required. We trained both a MobileNet (100 epochs, BS 64) and a ResNet18 (300 epochs, BS 128) with MSE. If not stated otherwise, the 300 epoch ResNet is the default target for reconstructon.

Using the labels downloaded from [here](https://github.com/DCGM/ffhq-features-dataset). The vector is made up of 32 elements, and contains:

* (4) img_data["faceRectangle"], 
* (3) img_data["faceAttributes"]["headPose"],
* (3) img_data["faceAttributes"]["facialHair"],
* (8) img_data["faceAttributes"]["emotion"],
* (2) img_data["faceAttributes"]["makeup"],
* (3) img_data["faceAttributes"]["occlusion"],
* (1) img_data["faceAttributes"]["smile"],         
* (1) img_data["faceAttributes"]["age"],         
* (1) img_data["faceAttributes"]["blur"]["value"],         
* (1) img_data["faceAttributes"]["exposure"]["value"],         
* (1) img_data["faceAttributes"]["noise"]["value"],         
* (1) img_data["faceAttributes"]["hair"]["bald"],         
* (1) float(img_data["faceAttributes"]["hair"]["invisible"]),
* (1) img_vector.append(float(img_data["faceAttributes"]["gender"] == "female"))
* (1) img_vector.append(GLASSES[img_data["faceAttributes"]["glasses"]])

Summarized:
* (7) for the face position
* (6) for the image (quality and occlusion)
* (5) for (facial) hair (note: the color was classified in the label set, but is not included here!)
* (9) for emotion
* (1) each for gender, age, glasses
* (2) for makeup