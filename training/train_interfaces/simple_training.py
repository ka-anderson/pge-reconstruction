# from training.logger import Logger
import torch
from .train_interface_base import TrainInterface
from ..logger import Logger

class ClassificationTrainInterface(TrainInterface):
    '''
    Simple default training:
    dataloader sample = X, y
    loss = lossfn(model(X), y)
    '''
    def __init__(self, optimizer, loss, model) -> None:
        super().__init__(optimizer, loss, model)
        self.options.update({"classname": "ClassificationTrainInterface"})

    def train(self, dataloader, logger:Logger, epoch):
        size = len(dataloader.dataset)
        self.model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)
            

            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.print_substep(logger, batch * len(X), size, {"loss": loss})
        
        return {"loss": loss.item()}
    
class AutoEncoderTrainInterface(TrainInterface):
    def __init__(self, optimizer, loss_fn, encoder, decoder) -> None:
        super().__init__(optimizer, loss_fn)

        self.encoder = encoder
        self.decoder = decoder

        self.options.update({
            "classname": "AutoEncoderTrainInterface",
            "encoder": encoder,
            "decoder": decoder,
        })

    def call_for_all_models(self, func):
        self.encoder = func(self.encoder)
        self.decoder = func(self.decoder)

    def get_learning_models(self):
        return {"encoder": self.encoder, "decoder": self.decoder}
    
    def get_model_out(self, input):
        return self.decoder(self.encoder(input))

    def train(self, dataloader, logger: Logger, epoch: int) -> dict:
        size = len(dataloader.dataset)
        self.encoder.train()
        self.decoder.train()
        for batch, (X, _) in enumerate(dataloader):
            X = X.to(self.device)

            coded = self.encoder(X)
            decoded = self.decoder(coded)
            loss = self.loss_fn(decoded, X)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.print_substep(logger, batch * len(X), size, {"loss": loss})
        
        return {"loss": loss.item()}