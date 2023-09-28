import torch
from datasets.dataset_base import CustomDataset
import os
from os.path import join
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist

from misc_helpers.helpers import repo_dir
from .logger import Logger
from .train_interfaces.train_interface_base import TrainInterface

def run_training(
    seed: int,
    exp_group: str,
    exp_id: str,
    epochs: int,
    train_interface: TrainInterface,
    dataset: CustomDataset,
    start_from_epoch: int=0, 
    use_training_data_to_train = True,
    gpus: list = [0],
    debug_distributed = False,
    tboard_model_weights: bool = False,
    tboard_model_outputs: bool = False,
    save_model: bool = False,
    save_model_freq = 0, 
    print_freq: int = 500,
    verbose_mode = False,
    use_tboard = False,
    batch_size_schedule = None
    ):
    '''
    * exp_group: parent folder the experiment
    * exp_id: name of this experiment. Results are saved into exp_group/output/exp_id
    * start_from_epoch: if the training process stopped at epoch n (loading model_n.pth), start_from_epoch would be n+1
    * debug_distributed: use the distributed method even for single gpu training
    * save_model_freq: save if batchstep % save_model_freq == 0
    * print_freq: print substep if epoch % print_freq == 0
    * batch_size_schedule: a list with tuples [..., (m_i, n_i), ...] indicating that step i runs for m_i epochs, using batch size n_i
    '''

    torch.set_printoptions(profile="full", linewidth=350)

    parameters = locals()
    folder = repo_dir("experiments", exp_group, "output", exp_id)

    logger = Logger(folder, substep_print_freq=print_freq, verbose=verbose_mode)
    logger.write_config(parameters)

    # torch.backends.cudnn.enabled = False

    if len(gpus) > 1 or debug_distributed:
        mp.spawn(_train_process_dist, 
                args=(len(gpus),                    
                    folder,
                    gpus, 
                    epochs, 
                    dataset, 
                    train_interface, 
                    save_model,
                    save_model_freq,
                    tboard_model_weights,
                    tboard_model_outputs,
                    logger, # the tboard writer of the logger must not be initialized before passing it to the process. For reasons.
                    verbose_mode,
                    use_tboard,
                    use_training_data_to_train,
                    start_from_epoch,
                    batch_size_schedule,
                    seed,
                    ),
                nprocs=len(gpus),
                join=True)
    else:
        _train_process(folder, 
                    epochs, 
                    dataset, 
                    train_interface,  
                    save_model,
                    save_model_freq,
                    tboard_model_weights,
                    tboard_model_outputs,
                    logger,
                    verbose_mode,
                    use_tboard,
                    use_training_data_to_train,
                    start_from_epoch,
                    batch_size_schedule,
                    seed,
                    gpus[0],
                    )

def _train_process(folder,
                    epochs, 
                    dataset: CustomDataset, 
                    train_interface: TrainInterface, 
                    save_model,
                    save_model_freq,
                    tboard_model_weights,
                    tboard_model_outputs,
                    logger:Logger,
                    verbose_mode,
                    use_tboard,
                    use_training_data_to_train,
                    start_from_epoch,
                    batch_size_schedule,
                    seed,
                    gpu,
    ):
    
    print("Initializing single-GPU training")

    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)

    if use_tboard:
        logger.init_tboard_writer()
    train_interface.to(gpu)

    save_model_weights(folder, -1, train_interface.get_learning_models())
    
    for e in range(start_from_epoch, epochs):
        batch_size = get_batchsize(batch_size_schedule, e)
        if dataset != None: # some experiments are not using a dataset
            if use_training_data_to_train:
                dataloader = dataset.get_train_dataloader(batch_size=batch_size)
            else:
                dataloader = dataset.get_test_dataloader(batch_size=batch_size, shuffle=True)
        else:
            dataloader = None

        train_metrics = train_interface.train(dataloader, logger, e)

        train_metrics = {f"trainI_{k}": v for k, v in train_metrics.items()}
        if batch_size != None:
            train_metrics.update({"batchsize": batch_size})
        logger.write_step(e, train_metrics)

        # (optional) save the partially trained model
        if save_model and save_model_freq != 0 and e % save_model_freq == 0:
            print(f"Trainer - saving model for epoch {e}.")
            save_model_weights(folder, e, train_interface.get_learning_models())


        # (optional) print the weights to tensorboard
        if tboard_model_weights:
            for name, model in train_interface.get_learning_models().items():
                logger.tboard_model_weights(model, name)

        # (optional) print output images to tensorboard
        if tboard_model_outputs:
            data = next(iter(dataloader))[0].to(gpu)
            logger.tboard_img_out(train_interface.get_model_out(data), "model_out")

    if save_model:
        save_model_weights(folder, epochs-1, train_interface.get_learning_models())

    print(f"Trainer done!")

def _train_process_dist(rank: int, world_size: int, 
                   folder,
                   gpus: list, 
                   epochs, 
                   dataset: CustomDataset, 
                   train_interface: TrainInterface, 
                   save_model,
                   save_model_freq,
                   tboard_model_weights,
                   tboard_model_outputs,
                   logger: Logger,
                   verbose_mode,
                   use_tboard,
                   use_training_data_to_train,
                   start_from_epoch,
                   batch_size_schedule,
                   seed,
                ):
        '''
        The process that is run on a single GPU
        '''

        # torch.autograd.set_detect_anomaly(True)

        # set seeds in every process (https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)

        is_main_process = (rank == 0)
        rank = gpus[rank]
        print(f"Initializing process {rank} (main process: {is_main_process})")

        if is_main_process and use_tboard:
            logger.init_tboard_writer()

        # setup
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

        train_interface.to(rank, distributed=True)

        for e in range(start_from_epoch, epochs):
            if dataset != None: # some experiments are not using a dataset
                batch_size = get_batchsize(batch_size_schedule, e)
                if use_training_data_to_train:
                    dataloader = dataset.get_train_dataloader(num_replicas=world_size, rank=rank, batch_size=batch_size)
                else:
                    dataloader = dataset.get_test_dataloader(num_replicas=world_size, rank=rank, batch_size=batch_size, shuffle=True)
            else:
                dataloader = None

            # train interface step, only pass the logger to the main gpu
            if is_main_process:
                train_metrics = train_interface.train(dataloader, logger, e)
            else:
                train_metrics = train_interface.train(dataloader, None, e)

            torch.distributed.barrier()

            # do this on only on one GPU
            if is_main_process:
                train_metrics = {f"trainI_{k}": v for k, v in train_metrics.items()}
                if batch_size != None:
                    train_metrics.update({"batchsize": batch_size})

                logger.write_step(e, train_metrics)

                # (optional) save the partially trained model
                if save_model and save_model_freq != 0 and e % save_model_freq == 0:
                    print(f"Trainer ({rank}) - saving model for epoch {e}.")
                    save_model_weights(folder, e, train_interface.get_learning_models())

                # (optional) print the weights to tensorboard
                if tboard_model_weights:
                    for name, model in train_interface.get_learning_models().items():
                        logger.tboard_model_weights(model, name)

                # (optional) print output images to tensorboard
                if tboard_model_outputs:
                    data = next(iter(dataloader))[0].to(rank)
                    logger.tboard_img_out(train_interface.get_model_out(data), "model_out")

        if is_main_process and save_model:
            save_model_weights(folder, epochs, train_interface.get_learning_models())

        # cleanup
        dist.destroy_process_group()
        print(f"Trainer ({rank}) done!")
    
def get_batchsize(batch_size_schedule, epoch):
    if batch_size_schedule == None:
        return None
        
    counted_epochs = 0
    for step, (num_e_in_step, batch_size) in enumerate(batch_size_schedule):
        interval_start = counted_epochs
        interval_stop = counted_epochs + num_e_in_step

        if epoch >= interval_start and epoch < interval_stop:
            return batch_size
        
        counted_epochs = interval_stop

    raise Exception("Number of epochs does not match the batch size schedule")


def save_model_weights(base_folder:str, epoch:int, models:dict):
    '''
    models: {model_name: model} (with model.state_dict returning its weights)
    path of the saves weights will be: {base_folder}/model_weights/{model_name}_epoch.pth
    '''
    if not os.path.exists(join(base_folder, "model_weights")):
        os.makedirs(join(base_folder, "model_weights"))
    for name, model in models.items():
        torch.save(model.state_dict(), join(base_folder, "model_weights", f'{name}_{epoch}.pth'))



