"""
WORK IN PROGRESS!!

The train function in train.py is becoming very messy. This file defines some classes in an attempt to 
resolve the issues that arise from both training and logging at the same time. 

"""

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader 
from typing import Callable
import wandb

class Logger(nn.Module):
    def __init__(self) -> None:
        super().__init__()

class Trainer(nn.Module):
    def __init__(
        self, 
        model: nn.Module, 
        data_pipeline_train: nn.Module, 
        data_pipeline_val: nn.Module, 
        model_saver, 
        criterion: Callable, 
        optimizer: torch.optim.Optimizer, 
        device: str, 
        verbose: bool=True, 
        validate_every: int=-1,
    ):
        super().__init__()
        self.model = model
        self.data_pipeline_train = data_pipeline_train
        self.data_pipeline_val = data_pipeline_val
        self.model_saver = model_saver
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.validate_every = validate_every
        self.verbose = verbose
        self.train_logger = TrainLogger()
    
    def to_device(self, d: dict):
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                d[k] = v.to(self.device)
        return d

    def forward_item(self, d: dict):
        """
        Get logits (model output) and labels from a dictionary d provided by a data loader
        """
        d = self.to_device(d)
        d = self.data_pipeline_train(d)
        x, y = d['x'], d['y'].float()
        logits = self.model(x)
        return logits, y

    def validate(
        self, 
        epoch_logger: Logger, 
        val_loader: DataLoader, 
        i: int
    ):
        """
        Perform validation and log it to the epoch_logger
        """
        
        with torch.no_grad():
            for d_v in val_loader:
                y_v_logits, y_v = self.forward_item(d_v)
                y_v_pred = torch.sigmoid(y_v_logits)
                epoch_logger.register_val(i, y_v_pred, y_v) 

        epoch_logger.val_report(i)

    def step(self, d):
        logits, y = self.forward_item(d)
        y_pred = torch.sigmoid(logits)
        self.optimizer.zero_grad()
        loss = self.criterion(y_pred, y)
        loss.backward()
        self.optimizer.step()
        return loss

    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        epochs: int=1,
    ):
        """
        Train for some epochs using the train and validation loaders
             
        """
        self.train_logger.start_training()
        validate_every = self.validate_every

        for epoch in range(epochs):
            epoch_logger = EpochLogger(epoch=epoch)
            self.train_logger.register(epoch_logger)

            for i, d in enumerate(train_loader):
                
                # optimization step
                loss = self.step(d)
                
                # reporting etc
                if i % validate_every == validate_every-1: 
                    self.validate(epoch_logger, val_loader, i)

            epoch_logger.train_update(loss)

            if validate_every == -1:
                self.validate(epoch_logger, val_loader, validate_every)
                
            if self.model_saver != None:
                self.model_saver.save_best_model(epoch_logger.epoch_val_loss, epoch, self.model, self.optimizer, self.criterion)

            epoch_logger.finish_epoch(train_loader) # do some final computations # TODO
            self.train_logger.finish_epoch() # report to wandb etc # TODO

        # At the end of training: Save model and training curve
        self.model_saver.save_final_model(epochs, self.model, self.optimizer, self.criterion) 
        self.model_saver.save_plots(train_loss, val_loss)
    

class EpochLogger(Logger):
    def __init__(self, epoch: int, trainer: Trainer) -> None:
        super().__init__()
        self.epoch = epoch 
        self.running_train_loss = 0.
        self.validation_loss = 0. 
        self.trainer = trainer
        self.val_buffer = {}
        self.val_reports = {}
        if trainer.verbose:
            print(f'starting epoch {epoch}')
    
    def __call__(self):
        pass 

    def report(
        self, 
        i, 
    ):
        """
        Report on the progress
        TODO: incorporate metrics here
        """
        print(
            f'''
                epoch {self.epoch+1}, 
                iteration {i}:\t
                running loss = {self.running_train_loss/(i+1):.3f}\t
                validation loss = {self.validation_loss:.3f}\t
            '''
        ) 

    def train_update(self, loss):
        self.running_train_loss += loss.item()
    
    def validation_update(self, loss):
        self.validation_loss += loss.item()
            
    def val_report(self, i: int): # TODO
        """
        Compute loss and metrics from trainer and self.val_buffer
        """
        loss_buffer = []
        metric_buffer = {m.name: [] for m in self.trainer.metrics} # TODO: implement metrics like this

        for pred, y in self.val_buffer[i]:
            for metric in self.trainer.metrics:
                metric_buffer[metric.name].append(metric(pred, y))
            loss_buffer.append(self.trainer.criterion(pred, y))
        
        metric_buffer['loss'] = loss_buffer
        
        # make everything tensors
        for k, v in metric_buffer.items():
            metric_buffer[k] = torch.concat(v)
        
        self.val_reports[i] = metric_buffer

        if self.trainer.verbose:
            print(f'iteration {i}')
            print(*[(k, v.mean()) for k, v in metric_buffer.items()], sep='\n') 

    def register_val(self, i, pred, y):
        """
        Register prediciton and ground truth for iteration i
        """
        if not i in self.val_buffer.keys():
            self.val_buffer[i] = [(pred, y)]
        else:
            self.val_buffer[i].append((pred, y))
    
    def finish_epoch(self, loader):
        if self.trainer.verbose:
            print(f'running loss: {self.running_train_loss / len(loader)}')

class WandbLogger():
    def __init__(self) -> None:
        super().__init__()
        pass

    def __call__(self):
        pass 

class TrainLogger(Logger):
    def __init__(self, trainer: Trainer, columns: list=['Sprectrogram', 'Predicted', 'Expected']) -> None:
        super().__init__()
        self.epochs = []
        self.trainer = trainer
        self.wandb_table = wandb.Table(columns=columns)
    
    def register(self, ep: EpochLogger):
        """
        Register an epoch logger
        """
        self.epochs.append(ep)
    
    def get_logs(self):
        return self.epochs 

    def start_training(self):
        if self.trainer.verbose:
            print(f'{20*"#"}\nStarting Training on {self.trainer.device} \n{20*"#"}')

    def finish_epoch(self):
        raise NotImplementedError

