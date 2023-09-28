FROM nvcr.io/nvidia/pytorch:23.04-py3


RUN apt update --allow-insecure-repositories
RUN apt install -y tmux
COPY misc_helpers/tmux.conf /etc/

ENV PYTHONPATH "${PYTHONPATH}:data/stylegan2-ada-pytorch/torch_utils:/data/stylegan2-ada-pytorch/dnnlib:/data/:/data/pge-reconstruction:/data/stylegan2-ada-pytorch"

COPY requirements.txt /data/
RUN pip install -r /data/requirements.txt

RUN echo "root:x" | chpasswd
ARG currUID
RUN useradd -m -u $currUID ka
USER ka

WORKDIR /data