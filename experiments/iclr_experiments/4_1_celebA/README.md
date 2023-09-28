# 4.1 Ensemble Reconstruction from CelebA trained encoders

This folder fuses the results of encoders trained in 3_1: the FN-trained P for block 2 and 3, the MSE-trained P for the rest. Note that the implementation of the StyleGAN allows feeding two different style vectors to the same block, resulting in a total fo 14 stylevectors.
