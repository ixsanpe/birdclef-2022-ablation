"""
Utility functions and classes for training
"""

import torch
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import os

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


def collate_fn(data: tuple):
    """
    Define how the DataLoaders should batch the data

    Here, we pad x and return a dict with x, y, and len, indicating what the 
    duration of the original x was, as this is not visible after padding but useful to know.

    Parameters:
        data:
            tuple such that data[0] is the audio signal to be processed and data[1] is the corresponding label
        device:
            'cpu' or 'cuda'; the device on which to train
    
    Returns:
        a dict of the padded input 'x', the label 'y' and the lengths 'lens' of the original input
        Every item in the dict is a torch.Tensor
    """
    max_dim = max([d[0].shape[-1] for d in data])
    pad_x = lambda x: torch.concat([x, torch.zeros((max_dim - x.shape[-1], ))])
    x = torch.stack([pad_x(d[0]) for d in data], axis=0)
    y = torch.stack([torch.tensor(d[1]) for d in data])
    lens = [d[0].shape[-1] for d in data]
    return {'x': x, 'y': y, 'lens': torch.tensor(lens)}