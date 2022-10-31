from torch.nn import functional as F
from torch import nn
import torch
from torch.distributions import Beta
from torch.nn.parameter import Parameter
import wandb


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
    val_metric: list=[]
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

    wandb.log({"train_loss": train_loss,
        "val_loss": val_loss,
        "val_metric" : val_metric})


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
