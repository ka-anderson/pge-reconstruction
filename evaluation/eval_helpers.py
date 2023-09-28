from matplotlib import pyplot as plt
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import pandas as pd
import torch

def plot_image_grid(images, output_path, ncols, nrows, figsize=None, norm_min=0, norm_max=1, pretty_plots=False):
    if figsize == None:
        figsize = (ncols, nrows)
    if pretty_plots:
        height, width, depth = next(iter(images.values())).shape
        dpi = 80
        figsize = (ncols * width / float(dpi), nrows * height / float(dpi))

    _, axs = plt.subplots(nrows, ncols, figsize=figsize)
    axs = axs.flatten()

    if isinstance(images, dict):
        images = {name: (value - norm_min)/(norm_max - norm_min) for name, value in images.items()}

        for (id, img), ax in zip(images.items(), axs):
            ax.imshow(img)
            if not pretty_plots:
                ax.set_title(id, fontsize=6)
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            if pretty_plots:
                ax.set_axis_off()
    else:
        images = (images - norm_min) / (norm_max - norm_min) 

        for img, ax in zip(images, axs):
            ax.imshow(img)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if pretty_plots:
                ax.set_axis_off()

    if pretty_plots:
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    else:
        plt.subplots_adjust(wspace=0, hspace=.5)
        plt.tight_layout()
    plt.savefig(join(output_path))
    plt.show()
    print(join(output_path))


def _torch_to_numpy_image(torch_image):
    '''
    move an image to the cpu and switch the channel dim, so that we get the default numpy structure (w, h, c)
    '''
    return torch_image.detach().cpu().permute(1, 2, 0)

def read_pd_markdown(path):
    # https://stackoverflow.com/questions/60154404/is-there-the-equivalent-of-to-markdown-to-read-data
    df = pd.read_table(path, sep="|", header=0, index_col=1, skipinitialspace=True).dropna(axis=1, how='all').iloc[1:] 
    df.columns = df.columns.str.rstrip()
    df.columns = df.columns.str.lstrip()

    df = df.apply(pd.to_numeric, errors='ignore')

    return df