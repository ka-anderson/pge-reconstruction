import pathlib
from os.path import join
import os

def repo_dir(*paths):
    # find the difference between the cwd (where the console is) and the parentfolder of this file (the root of the project) in case the console/cwd is in a parent dir of the project

    path = str(pathlib.Path(__file__).parent.resolve()).split("/")
    cwd = os.getcwd().split("/")[-1]
    cwd_index = path.index(cwd)
    
    path = path[cwd_index + 1:-1]

    return join(*path + list(paths))