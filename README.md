# Revealing Unintentional Information Leakage in Low-Dimensional Facial Portrait Representations
*Kathleen Anderson and Thomas Martinetz, ICANN 2024*

We evaluate the information that can unintentionally leak into the low dimensional output of a neural network, by reconstructing an input image from a 40- or 32-element feature vector that intends to only describe abstract attributes of a facial portrait. The reconstruction only uses blackbox-access to the image encoder which generates the feature vector. Other than previous work, we leverage recent knowledge about image generation and facial similarity, implementing a method that outperforms the current state-of-the-art. Our strategy uses the pretrained StyleGAN and a new loss function that compares the perceptual similarity of portraits by mapping them into the latent space of a FaceNet embedding. Additionally, we present a new technique which fuses the output of an ensemble, to deliberately generate specific aspects of the recreated image. 

Please be aware that is a portion of our actual research code, which is made not as a lightweight library, but as a flexible research environment.

To run the experiments in the exact same manner as we did:
* download the code for [StyleGAN2](https://github.com/NVlabs/stylegan2-ada-pytorch) and [DiffAugGANS](https://github.com/mit-han-lab/data-efficient-gans) and place them (for example) in the same parent directory as this repository
* download a dataset e.g. [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) or [FFHQ](https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq) into the datasets/downloads folder. FFHQ expects the images in datasets/downloads/ffhq/org/0, CelebA expects the entire downloaded folder in datasets/downloads/celeba/
* download a pretrained image generator. The default training expects 'stylegan_ffhq_256.pkl' in models/pretrained, downloaded from [here](https://github.com/mit-han-lab/data-efficient-gans/tree/master/DiffAugment-stylegan2) and translated using
```
python data-efficient-gans/DiffAugment-stylegan2-pytorch/legacy.py --source=pge-reconstruction/models/pretrained/stylegan_ffhq_256_tf.pkl --dest=pge-reconstruction/models/pretrained/stylegan_ffhq_256.pkl
```
* build and launch the docker container
```
docker build --build-arg currUID=$(id -u) -t pgereconstruction:latest -f misc_helpers/Dockerfile .
docker run -dit --shm-size=2G --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -v .:/data/ pgereconstruction
``` 
the "-v ." might have to be replaced with the absolute path to the parent directory of this repository and the two additional repositories.
* From inside the container, run one of the train_x_y.py scripts. 
