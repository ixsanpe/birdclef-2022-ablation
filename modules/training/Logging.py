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

        buf = self.val_buffer[i]
        pred = torch.concat([b[0] for b in buf], axis=0)
        y = torch.concat([b[-1] for b in buf], axis=0)

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

class WandbLogger(Logger):
    def __init__(
        self, 
        trainer, 
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
        trainer, 
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
      