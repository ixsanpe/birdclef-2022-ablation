from torch.nn import functional as F
from torch import nn
import torch
from torch.distributions import Beta
from torch.nn.parameter import Parameter
import wandb
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import os


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)
        return ret

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class Mixup(nn.Module):
    def __init__(self, mix_beta):

        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)

    def forward(self, X, Y, weight=None):

        bs = X.shape[0]
        n_dims = len(X.shape)
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)

        if n_dims == 2:
            X = coeffs.view(-1, 1) * X + (1 - coeffs.view(-1, 1)) * X[perm]
        elif n_dims == 3:
            X = coeffs.view(-1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1)) * X[perm]
        else:
            X = coeffs.view(-1, 1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1, 1)) * X[perm]

        Y = coeffs.view(-1, 1) * Y + (1 - coeffs.view(-1, 1)) * Y[perm]

        if weight is None:
            return X, Y
        else:
            weight = coeffs.view(-1) * weight + (1 - coeffs.view(-1)) * weight[perm]
            return X, Y, weight


def print_probability_ranking(y, n=5):
    assert n <= len(y)

    y = nn.functional.sigmoid(y)

    output = ""
    sorted, indices = torch.sort(y, descending=True)

    for i in range(n):
        output += "#%i   Class: %i   Prob: %.3f\n"%(i, indices[i], sorted[i])

    return output


def wandb_log_stats(
    train_loss: list = [], 
    val_loss: list=[], 
    val_metrics: list=[]
):
    """
    sends the current training statistic to Weights and Biases
    Parameters:
        train_loss:
            training loss, evaluated by cost function on training set
        val_loss: 
            validation loss, evaluated by cost function on validation set
        val_metric: 
            validation metric, evaluated by metric function on validation set
    """
    """
    print(val_metrics)
    val_metrics_new = {}
    for el in val_metrics:
        for key, object in el.items():
            if not val_metrics_new[key]:
                val_metrics_new[key] = []
            val_metrics_new[key].append(object)
    # transform list of dict to dict of list
    """
    log_dict = {"train_loss": train_loss,
        "val_loss": val_loss
        }
    log_dict.update(val_metrics)
    wandb.log(log_dict)


def wandb_log_spectrogram(
    model,
    data_pipeline_val,
    val_loader,
    device, 
    wandb_spec_table,
    n_splits
):
    """
    Runs model on validation set and sends the first n batches of spectrogram, label, and prediction to Weights and Biases
    Parameters:
        model:
            model for which to report progress
        data_pipeline_val: 
            Pre-processing pipeline of the validation data from audio signal to model input
            data_pipeline_val should take a tuple (x, y) as an input
        val_loader: 
            data loader for validation data
        device: 
            device on which to train model
        wandb_spec_data: 
            The wandb table to save the data to
        n_splits:
            number of splits of 30s recordings

    """
    with torch.no_grad():
        # only takes the first n batches
        n = 1 # maximum amount of batches to evaluate

        for i, (x_v,y_v) in enumerate(val_loader): 
            x_v, y_v = data_pipeline_val((x_v.to(device), y_v.to(device).float()))
            y_v_pred = model(x_v)
            #print(y_v_pred.shape, x_v.shape, y_v.shape)
            
            for j, x_v_slice in enumerate(x_v):
                y_v_slice, y_v_slice_pred = y_v[int(j/n_splits)], y_v_pred[int(j/n_splits)]
                wandb_spec_table.add_data(wandb.Image(x_v_slice), print_probability_ranking(y_v_slice), print_probability_ranking(y_v_slice_pred))
            if i+1 >= n:
                break

            # if we want to leverage predictions, we can do this instead:
            """
            for i, y_v_slice in enumerate(y_v):
                grid_image = make_grid(x_v.unsqueeze(1)[i*n_splits:(i+1)*n_splits], nrow=1)[0,...]
                y_v_slice = leverage(y_v_pred[i*n_splits:(i+1)*n_splits]) # leverage has to be implemented
                wandb_spec_table.add_data(wandb.Image(grid_image), print_probability_ranking(y_v_slice), print_probability_ranking(y_v_slice_pred))
            """

        wandb.log({"predictions": wandb_spec_table})



class ModelSaver:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, save_dir, name, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        self.save_dir = save_dir
        self.name = name

        if not os.path.exists(self.save_dir):
            "Warning: Save dir %s does not exist. Trying to create dir..."%(self.save_dir)
            os.mkdir(save_dir)
        
    def save_best_model(
        self, current_valid_loss, 
        epoch, model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer' : optimizer,
                'loss': criterion,
                }, '%s/%s_best_model.pth'%(self.save_dir, self.name))


    def save_final_model(self, epochs, model, optimizer, criterion):
        """
        Function to save the trained model to disk.
        """
        print(f"Saving final model...")
        torch.save({
                    'epoch': epochs,
                    'model_state_dict': model.state_dict(),
                    'optimizer' : optimizer,
                    'loss': criterion,
                    }, '%s/%s_final_model.pth'%(self.save_dir, self.name))

    def save_plots(self, train_loss, valid_loss, train_metric = [], val_metric = []):
        """
        Function to save the loss and accuracy plots to disk.
        """

        # loss plots
        plt.figure(figsize=(10, 7))
        plt.plot(
            train_loss, color='orange', linestyle='-', 
            label='train loss'
        )
        plt.plot(
            valid_loss, color='red', linestyle='-', 
            label='validataion loss'
        )
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('%s/%s_loss.png'%(self.save_dir, self.name), bbox_inches='tight')


        if train_metric or val_metric:
            
            plt.figure(figsize=(10, 7))
            if train_metric:
                plt.plot(
                    train_metric, color='green', linestyle='-', 
                    label='train accuracy'
                )
            if val_metric:
                plt.plot(
                    val_metric, color='blue', linestyle='-', 
                    label='validataion accuracy'
                )
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig('%s/%s_metric.png'%(self.save_dir, self.name), bbox_inches='tight')

def print_output(
    train_loss :float = 0., 
    current_loss: float = 0.,
    train_metrics :dict = None, 
    val_loss :float = 0., 
    val_metrics :dict=None,
    i: int=1,
    max_i: int=1,
    epoch :int = 0):
    print(f'epoch {epoch+1}, iteration {i}/{max_i}:\trunning loss = {train_loss:.3f}\tcurrent loss = {current_loss:.3f}\tvalidation loss = {val_loss:.3f}' + print_metrics(train_metrics)+ print_metrics(val_metrics)) 


def print_metrics(
    metrics: dict,
    prefix: str = ''
):

    """
    function to print out the dict of metrices
    metrics:
        a dict of metrices with the form {'metric_name': metric_func}
    """
    output = ""

    if metrics != None:
        for me_name, me_score in metrics.items():
            output = output + f"\t {prefix}{me_name} = {me_score:.3f}"

    return output