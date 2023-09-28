from torch.nn.parallel import DistributedDataParallel as DDP
from ..logger import Logger

class TrainInterface:
    def __init__(self, optimizer=None, loss_fn=None, model=None) -> None:
        '''
        Use this to add the default inputs to the interface: a single optimizer, loss_fn, and model
        '''
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.model = model
        self.options = {
            "optimizer": optimizer,
            "loss_fn": loss_fn,
            "model": model,
        }

    def call_for_all_models(self, func):
        '''
        call a function on every model (even those that are not learning):
            model = func(model)

        (the func needs to return the model!)
        '''
        self.model = func(self.model)
    
    def call_for_all_learning_models(self, func):
        '''
        same as call_for_all_models, but only for models that require grad (used for DDP)
        '''
        self.call_for_all_models(func)

    def to(self, device, distributed=False, find_unused_parameters=False):
        print(f"TrainInterface - moving all models to device {device} (distributed: {distributed})")
        def move(model):
            model = model.to(device)
            return model
        def create_ddp(model):
            model = DDP(model, device_ids=[device], output_device=device, find_unused_parameters=find_unused_parameters)
            return model

        self.device = device
        self.call_for_all_models(move)

        if distributed:
            self.call_for_all_learning_models(create_ddp)

    def train(self, dataloader, logger:Logger, epoch: int) -> dict:
        '''
        use the logger to print_substep(batch, total, metrics)
        return a dict containing epoch metrics (like loss)
        '''
        raise NotImplementedError()
    
    @classmethod
    def load_from_folder(cls, folder_path, epoch):
        '''
        Load the interface from a given epoch.
        matching .pts files need to be saved in folder_path/model_weights
        Currently not implemented for every training interface
        '''
        raise NotImplementedError()

    def get_learning_models(self):
        '''
        return all models that are part of the training, as dict: {model_name: model}
        (only models whose weights are changing)
        used to log model weights
        '''
        return {"model": self.model}

    def get_model_out(self, input):
        '''
        return a batch of model outputs, passes one batch of the input dataset, in case that is required.
        '''
        return self.model(input)
    
    def print_substep(self, logger:Logger, batch, total, metrics):
        if logger != None and logger.verbose:
            logger.print_substep(batch, total, metrics)