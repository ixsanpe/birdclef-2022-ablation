"""
During training and validation we perform logging both to the console and to wandb
This script defines the relevant loggers, keeping track of the metrics and 
reporting the performance during training and validation. 
"""

from torch import nn 
import torch 
import wandb
import time 
from modules import print_probability_ranking
from .Metric import Metric

class Logger(nn.Module):
    def __init__(self) -> None:
        super().__init__()

class EpochLogger(Logger):
    """
    This Logger keeps track of metrics and predictions during training and validation.
    As such, some of the functions are currently near duplicates of each other. 
    We decided to split these functions into training and validation versions so
    that we have better control over exactly what we would like to track for training
    and for validation. 
    """
    def __init__(self, epoch: int, trainer, metrics: list[Metric]) -> None:
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
        self.i = 0
        self.trainer = trainer
        self.val_buffer = {}
        self.train_buffer = {}
        self.val_reports = {}
        self.train_reports = {}
        if trainer.verbose:
            print(f'starting epoch {epoch}')

    def train_report(self):
        print(f'iteration {self.i}\t runnning loss {self.running_train_loss / self.i :.3f}\n')

    def train_report_metrics(self, i:int):
        """
        Compute metrics from trainer and self.val_buffer and save the relevant information
        The loss is not computed and it is kept as before
        """
        metric_buffer_train = {m.name: [] for m in self.metrics}

        buf = self.train_buffer[i]
        pred = torch.concat([b[0] for b in buf], axis=0)
        y = torch.concat([b[-1] for b in buf], axis=0) # ground truth

        for metric in self.metrics:
            metric_buffer_train[metric.name].append(metric(pred, y))

        # make everything tensors and keep only the mean
        for k, v in metric_buffer_train.items():
            metric_buffer_train[k] = torch.tensor(v).mean()

        self.train_reports[i] = metric_buffer_train  # save the buffer for later use

    def train_update(self, loss):
        """
        Update training progression. Currently only tracking loss
        """
        self.i = self.i + 1
        self.running_train_loss += loss.item()
            
    def val_report(self, i: int): 
        """
        Compute loss and metrics from trainer and self.val_buffer and save the relevant information
        """
        loss_buffer = []
        metric_buffer = {m.name: [] for m in self.metrics}

        buf = self.val_buffer[i]
        pred = torch.concat([b[0] for b in buf], axis=0)
        y = torch.concat([b[-1] for b in buf], axis=0) # ground truth

        for metric in self.metrics:
            metric_buffer[metric.name].append(metric(pred, y))
        loss_buffer.append(self.trainer.criterion(pred.double(), y.double()))
        
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
        Register prediciton and ground truth for iteration i in the validation buffer
        """
        assert torch.all(torch.logical_and(pred >= 0, pred <= 1)), f'got an invalid range for predictions: {pred.min()=}, {pred.max()=}'
        if not i in self.val_buffer.keys():
            self.val_buffer[i] = [(pred, y)]
        else:
            self.val_buffer[i].append((pred, y))

    def register_train(self, i, pred, y):
        """
        Register prediciton and ground truth for iteration i in the train buffer
        """
        assert torch.all(torch.logical_and(pred >= 0, pred <= 1)), f'got an invalid range for predictions: {pred.min()=}, {pred.max()=}'
        if not i in self.train_buffer.keys():
            self.train_buffer[i] = [(pred, y)]
        else:
            self.train_buffer[i].append((pred, y))

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

    def get_metric_train(self, name, i=-1):
        i = self.map_id_to_iteration_train(i)
        return self.train_reports[i][name]

    def get_train_buffer(self, i=-1):
        """
        Get the buffer containing predictions (in the interval (0, 1)) and ground truth for the ith validation run
        """
        i = self.map_id_to_iteration_train(i)
        return self.train_buffer[i]

    def map_id_to_iteration_train(self, i):
        """
        Map an index i to the corresponding iteration. For example, if we updated in iterations 0, 9, 19, then
        calling this method with i = 0, 1, -1 returns 0, 9, 19, respectively. Useful for searching dictionaries
        using indices.
        """
        return list(self.train_buffer.keys())[i]

    def get_metric_train(self, name, i=-1):
        i = self.map_id_to_iteration_train(i)
        return self.train_reports[i][name]
    
    def finish_epoch(self, loader):
        """
        do some final computations/reporting at the end of the epoch
        """
        assert self.i == len(loader)
        if self.trainer.verbose:
            print(f'finished epoch {self.epoch} with running loss: {self.running_train_loss/self.i :.3f}\n')

class WandbLogger(Logger):
    def __init__(
        self, 
        trainer, 
        stat_names: list[str]=['train_loss', 'val_loss'], 
        columns: list=['Predicted', 'Expected'], 
        project_name: str='Baseline', 
        experiment_name: str=None, 
        config: dict={}, 
        group=None,
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
        if group is None: group = experiment_name
        wandb.init(project=project_name, entity="ai4goodbirdclef", name=experiment_name, config=config, group=group)
        wandb.watch(trainer.model)

    def __call__(self, stats: dict, log_rankings=True):
        """
        stats is a dict with keys 'train_loss', 'val_loss', 'pred_ranking' and 'y_ranking'
        where train_loss, val_loss are floats and pred_ranking, y_ranking are iterable such that in each
        iteration the object returned is a probability ranking vector
        
        """
        '''
        # wandb.log({k: stats[k] for k in self.stat_names})
        for pred, y in zip(stats['pred_ranking'], stats['y_ranking']):
            self.wandb_table.add_data(pred, y)
        stats['predictions'] = self.wandb_table
        stats.pop('y_ranking')
        stats.pop('pred_ranking')

        # wandb.log({"predictions": self.wandb_table})
        wandb.log(stats)
        '''
        # wandb.log({k: stats[k] for k in self.stat_names})
        if log_rankings:
            for pred, y in zip(stats['pred_ranking_val'], stats['y_ranking_val']):
                self.wandb_table.add_data(pred, y)
            stats['predictions_val'] = self.wandb_table
            for pred, y in zip(stats['pred_ranking_train'], stats['y_ranking_train']):
                self.wandb_table.add_data(pred, y)
            stats['predictions_train'] = self.wandb_table
        try:
            stats.pop('y_ranking_val')
            stats.pop('y_ranking_train')
            stats.pop('pred_ranking_val')
            stats.pop('pred_ranking_train')
        except:
            pass 
        # wandb.log({"predictions": self.wandb_table})
        wandb.log(stats)
    
    def finish_run(self):
        wandb.finish()

class TrainLogger(Logger):
    def __init__(
        self, 
        trainer, 
        metrics: list[Metric], 
        keep_epochs=False, 
        use_wandb=True,
        columns: list=['Predicted', 'Expected'], 
        project_name: str='Baseline', 
        experiment_name: str=None, 
        config: dict={}, 
        group=None, 
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
                stat_names=['train_loss', 'val_loss'] + ['val_' + m.name for m in metrics] + ['train_'+ m.name for m in metrics],
                columns=columns,
                project_name=project_name,
                experiment_name=experiment_name,
                config=config, 
                group=group, 
            )
            
    
    def register(self, ep: EpochLogger):
        """
        Register an epoch logger
        """
        self.epochs.append(ep)
    
    def get_logs(self):
        """
        Return the epochs registered here 
        """
        return self.epochs 

    def start_training(self):
        if self.trainer.verbose:
            print(f'{20*"#"}\nStarting Training on {self.trainer.device} \n{20*"#"}')

    def start_epoch(self, epoch):
        epoch_logger = EpochLogger(epoch, self.trainer, self.metrics)
        self.register(epoch_logger)
        return epoch_logger

    def wandb_report(self):
        """
        Make a call to self.wandb
        """
        last_epoch = self.epochs[-1]
        buffer_val = last_epoch.get_validation_buffer()
        buffer_train = last_epoch.get_train_buffer()
        if self.use_wandb:
            # Make tensors of shape (num_validation_datapoints, num_classes) to rank probabilities
            preds = torch.concat([b[0] for b in buffer_val], axis=0)
            ys = torch.concat([b[-1] for b in buffer_val], axis=0)
            pred_ranking_val = [print_probability_ranking(pred) for pred in preds]
            y_ranking_val = [print_probability_ranking(y) for y in ys]
            preds = torch.concat([b[0] for b in buffer_train], axis=0)
            ys = torch.concat([b[-1] for b in buffer_train], axis=0)
            pred_ranking_train = [print_probability_ranking(pred) for pred in preds]
            y_ranking_train = [print_probability_ranking(y) for y in ys]
            stats = {
                'train_loss': last_epoch.running_train_loss / last_epoch.i, 
                'val_loss': last_epoch.validation_loss(), 
                'pred_ranking_val': pred_ranking_val,
                'y_ranking_val': y_ranking_val,
                'pred_ranking_train': pred_ranking_train,
                'y_ranking_train': y_ranking_train,
            }
            for m in self.metrics:
                stats['val_' + m.name] = last_epoch.get_metric(m.name)
                stats['train_' + m.name] = last_epoch.get_metric_train(m.name)
            self.wandb_logger(stats)
    
    def finish_epoch(self):
        """
        Manage epochs at the end of the run
        """
        if not self.keep_epochs:
            last_epoch = self.epochs[-1]
            del last_epoch # free up some memory

    def finish_run(self):
        self.wandb_logger.finish_run()
      