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
import time 
from modules.model_utils import print_probability_ranking

class Logger(nn.Module):
    def __init__(self) -> None:
        super().__init__()

class Metric():
    def __init__(self, name: str, metric: Callable):
        """
        A class that can compute a metric and also maintains a name for that metric
        """
        super().__init__()
        self.name = name 
        self.metric = metric 
    
    def __call__(self, x, y):
        return self.metric(x, y)

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
        metrics: list[Metric],
        keep_epochs: bool=False,
        verbose: bool=True, 
        use_wandb: bool=True,
        validate_every: int=-1,
        wandb_args={}, 
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
        self.train_logger = TrainLogger(self, metrics=metrics, use_wandb=use_wandb, keep_epochs=keep_epochs, **wandb_args)
    
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
            # epoch_logger = EpochLogger(epoch=epoch, trainer=self)
            # self.train_logger.register(epoch_logger)
            epoch_logger = self.train_logger.start_epoch(epoch)
            

            for i, d in enumerate(train_loader):
                
                # optimization step
                loss = self.step(d)
                epoch_logger.train_update(loss)

                # reporting etc
                if i % validate_every == validate_every-1: 
                    epoch_logger.train_report(len(val_loader))
                    self.validate(epoch_logger, val_loader, i)

            if validate_every == -1 or i < validate_every + 1:
                epoch_logger.train_report(len(val_loader))
                self.validate(epoch_logger, val_loader, validate_every)
                
            if self.model_saver != None:
                self.model_saver.save_best_model(epoch_logger.validation_loss(), epoch, self.model, self.optimizer, self.criterion)

            epoch_logger.finish_epoch(train_loader) # report final loss for this epoch
            self.train_logger.finish_epoch() # report to wandb etc 

        # At the end of training: Save model and training curve
        # self.model_saver.save_final_model(epochs, self.model, self.optimizer, self.criterion) TODO
        # self.model_saver.save_plots(train_loss, val_loss)

class EpochLogger(Logger):
    def __init__(self, epoch: int, trainer: Trainer, metrics: list[Metric]) -> None:
        """
        Keep track of different stats for an epoch

        Parameters:
            epoch:
                the epoch index to keep track of
            trainer:
                the Trainer being trained this epoch. EpochLogger uses many of trainer's attributes (e.g. trainer.verbose)
        
        """
        super().__init__()
        self.epoch = epoch 
        self.metrics = metrics
        self.running_train_loss = 0.
        self.trainer = trainer
        self.val_buffer = {}
        self.val_reports = {}
        if trainer.verbose:
            print(f'starting epoch {epoch}')

    def train_report(self, i):
        print(f'iteration {i}\t runnning loss {self.running_train_loss / (i+1) :.3f}\n')

    def train_update(self, loss):
        """
        Update training progression. Currently only tracking loss
        """
        self.running_train_loss += loss.item()
            
    def val_report(self, i: int): 
        """
        Compute loss and metrics from trainer and self.val_buffer and save the relevant information
        """
        loss_buffer = []
        metric_buffer = {m.name: [] for m in self.metrics}

        for pred, y in self.val_buffer[i]:
            for metric in self.metrics:
                metric_buffer[metric.name].append(metric(pred, y))
            loss_buffer.append(self.trainer.criterion(pred, y))
        
        metric_buffer['loss'] = loss_buffer
        
        # make everything tensors and keep only the mean
        for k, v in metric_buffer.items():
            metric_buffer[k] = torch.tensor(v).mean()
        
        self.val_reports[i] = metric_buffer # save the buffer for later use

        if self.trainer.verbose:
            print(f'validation report for iteration {i}')
            print(*[f'{k}\t {v: .3f}' for k, v in metric_buffer.items()], sep='\n') 

    def register_val(self, i, pred, y):
        """
        Register prediciton and ground truth for iteration i
        """
        assert torch.all(torch.logical_and(pred >= 0, pred <= 1)), f'got an invalid range for predictions: {pred.min()=}, {pred.max()=}'
        if not i in self.val_buffer.keys():
            self.val_buffer[i] = [(pred, y)]
        else:
            self.val_buffer[i].append((pred, y))
    
    def validation_loss(self, i=-1):
        """
        Return the mean loss of the ith validation run (i.e. a float)
        """
        return self.get_validation_report(i)['loss']

    def get_validation_report(self, i=-1):
        """
        Return the loss and metrics for the ith validation run
        """
        i = self.map_id_to_iteration(i)
        return self.val_reports[i]

    def get_validation_buffer(self, i=-1):
        """
        Get the buffer containing predictions (in the interval (0, 1)) and ground truth for the ith validation run
        """
        i = self.map_id_to_iteration(i)
        return self.val_buffer[i]

    def map_id_to_iteration(self, i):
        """
        Map an index i to the corresponding iteration. For example, if we updated in iterations 0, 9, 19, then
        calling this method with i = 0, 1, -1 returns 0, 9, 19, respectively. Useful for searching dictionaries
        using indices. 
        """
        return list(self.val_buffer.keys())[i]

    def get_metric(self, name, i=-1):
        i = self.map_id_to_iteration(i)
        return self.val_reports[i][name]
    
    def finish_epoch(self, loader):
        """
        do some final computations/reporting at the end of the epoch
        """
        self.running_train_loss = self.running_train_loss / len(loader)
        if self.trainer.verbose:
            print(f'finished epoch {self.epoch} with running loss: {self.running_train_loss :.3f}\n')

class WandbLogger():
    def __init__(
        self, 
        trainer: Trainer, 
        stat_names: list[str]=['train_loss', 'val_loss'], 
        columns: list=['Predicted', 'Expected'], 
        project_name: str='Baseline', 
        experiment_name: str=None, 
        config: dict={}
    ):
        """
        Parameters:
            trainer:
                the trainer to log
            columns:
                the columns for the wandb table we use
            prioject_name:
                name of the project
            experiment_name:
                name of the experiment
            config:
                config passed to wandb
        """
        super().__init__()
        if experiment_name is None:
            experiment_name = "baseline_" + str(int(time.time()))
        self.stat_names = stat_names
        self.wandb_table = wandb.Table(columns=columns)
        wandb.init(project=project_name, entity="ai4goodbirdclef", name=experiment_name, config=config)
        wandb.watch(trainer.model)

    def __call__(self, stats: dict):
        """
        stats is a dict with keys 'train_loss', 'val_loss', 'pred_ranking' and 'y_ranking'
        where train_loss, val_loss are floats and pred_ranking, y_ranking are iterable such that in each
        iteration the object returned is a probability ranking vector
        
        """
        wandb.log({k: stats[k] for k in self.stat_names})
        for pred, y in zip(stats['pred_ranking'], stats['y_ranking']):
            self.wandb_table.add_data(pred, y)
        wandb.log({"predictions": self.wandb_table})

class TrainLogger(Logger):
    def __init__(
        self, 
        trainer: Trainer, 
        metrics: list[Metric], 
        keep_epochs=False, 
        use_wandb=True,
        columns: list=['Predicted', 'Expected'], 
        project_name: str='Baseline', 
        experiment_name: str=None, 
        config: dict={}
    ):
        """
        Parameters:
            trainer:
                the trainer to keep track of
            keep_epochs:
                Whether to keep the epochs. This could cost memory which might be unnecessary
            use_wandb:
                whether to use Weights and Biases
            columns, project_name, experiment_name, config:
                wandb parameters if use_wandb
        """
        super().__init__()
        self.epochs = []
        self.trainer = trainer
        self.metrics = metrics
        self.keep_epochs = keep_epochs
        self.use_wandb = use_wandb
        if use_wandb:
            self.wandb_logger = WandbLogger(
                trainer, 
                stat_names=['train_loss', 'val_loss'] + [m.name for m in metrics], 
                columns=columns,
                project_name=project_name,
                experiment_name=experiment_name,
                config=config
            )
            
    
    def register(self, ep: EpochLogger):
        """
        Register an epoch logger
        """
        self.epochs.append(ep)
    
    def get_logs(self):
        """
        Return the epochs registered here 

        TODO: may want to change this because of memory issues
        One way to do so is to settle on some key statistics that we are interested in for each 
        epoch and only save those rather than all of the epoch data
        """
        return self.epochs 

    def start_training(self):
        if self.trainer.verbose:
            print(f'{20*"#"}\nStarting Training on {self.trainer.device} \n{20*"#"}')

    def start_epoch(self, epoch):
        epoch_logger = EpochLogger(epoch, self.trainer, self.metrics)
        self.register(epoch_logger)
        return epoch_logger

    def finish_epoch(self):
        """
        Make a call to self.wandb and manage epochs
        """
        last_epoch = self.epochs[-1]
        buffer = last_epoch.get_validation_buffer()

        # Make tensors of shape (num_validation_datapoints, num_classes) to rank probabilities
        preds = torch.concat([b[0] for b in buffer], axis=0)
        ys = torch.concat([b[-1] for b in buffer], axis=0)
        pred_ranking = [print_probability_ranking(pred) for pred in preds]
        y_ranking = [print_probability_ranking(y) for y in ys]
        if self.use_wandb:
            stats = {
                'train_loss': last_epoch.running_train_loss, 
                'val_loss': last_epoch.validation_loss(), 
                'pred_ranking': pred_ranking, 
                'y_ranking': y_ranking
            }
            for m in self.metrics:
                stats[m.name] = last_epoch.get_metric(m.name)
            self.wandb_logger(stats)
        if not self.keep_epochs:
            del last_epoch # free up some memory
        

