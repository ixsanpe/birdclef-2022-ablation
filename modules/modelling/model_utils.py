"""
Utility functions which can be used for modelling.
In our final report, we only use ModelSaver to save the best models
"""


from torch.nn import functional as F
from torch import nn
import torch
from torch.distributions import Beta
from torch.nn.parameter import Parameter
import warnings
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import os


def gem(x, p=3, eps=1e-6):
    """
    Taken from Henkel et. al. 
    https://github.com/ChristofHenkel/kaggle-birdclef2021-2nd-place/tree/26438069466242e9154aacb9818926dba7ddc7f0
    """
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    """
    Taken from Henkel et. al. 
    https://github.com/ChristofHenkel/kaggle-birdclef2021-2nd-place/tree/26438069466242e9154aacb9818926dba7ddc7f0
    """
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
    """
    Taken from Henkel et. al. 
    https://github.com/ChristofHenkel/kaggle-birdclef2021-2nd-place/tree/26438069466242e9154aacb9818926dba7ddc7f0
    """
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
    # sanity checks which can be used in weights and biases if desired
    assert n <= len(y)
    if isinstance(y, torch.Tensor) and torch.any(torch.logical_or(y>1, y<0)): 
        warnings.warn(f'WARNING! Got invalid range for y! \n{y.max()=}, \n{y.min()=}')

    output = ""
    sorted, indices = torch.sort(y, descending=True)

    for i in range(n):
        output += "#%i   Class: %i   Prob: %.3f\n"%(i, indices[i], sorted[i])

    return output


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
            warnings.warn("Warning: Save dir %s does not exist. Trying to create dir..."%(self.save_dir))
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
