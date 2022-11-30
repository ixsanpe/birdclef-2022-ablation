#%%
from modules import *

from sklearn import model_selection
import pandas as pd
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split

######################## STILL IN WORK ###############################
#%%
class CustomSplit():

    def __init__(self, 
        train_val_test_size: tuple=(0.9,0.05,0.05),
        split_type: str="random",
        random_state: int = None,
        num_classes = 152
        ) -> None:

        self.text_index = []
        self.train_index = []
        self.val_index = []

        self.train_size = train_val_test_size(0)
        self.val_size = train_val_test_size(1)
        self.test_size = train_val_test_size(2)

        if self.train_size + self.val_size + self.test_size != 1:
            print('[TrainTestSplit] Train_Test_Val ratio does not sum to 1. Scaling to 1....')
            ratio_sum = self.train_size + self.val_size + self.test_size
            self.train_size = train_val_test_size(0)/ratio_sum
            self.val_size = train_val_test_size(1)/ratio_sum
            self.test_size = train_val_test_size(2)/ratio_sum

        self.random_state = random_state
        self.num_classes = num_classes

        self.split_type = split_type
        self.implemented_splits = ['random', 'stratified']
        if self.split_type not in self.implemented_splits:
            print('[TrainTestSplit] Split type not implemented. Choosing random split')
            raise NotImplementedError("[TrainTestSplit] Split type {self.split_type} not implemented")

    def __call__(self, df:pd.DataFrame) -> tuple:

        def one_hot(labels: list):
            """
            Return a list of one-hot encodings for passed labels
            Parameters
                labels: 
                    list of labels to one-hot encode
            Returns:
                one-hot encoded labels
            """
            num2vec = lambda lbl: (np.arange(self.n_classes) == lbl).astype(int)
            if isinstance(labels[0], list):
                return [np.sum([num2vec(l) for l in lbls], axis=0) for lbls in labels]
            return [num2vec(lbl) for lbl in labels]

        if self.split_type == 'stratified':
            y = 


            
            







#%%
from decouple import config
SPLIT_PATH = config("SPLIT_PATH")
df_train = pd.read_csv(f'{SPLIT_PATH}train_metadata.csv')
# %%
