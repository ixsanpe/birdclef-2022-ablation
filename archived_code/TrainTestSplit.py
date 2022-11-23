from paths import *

from sklearn import model_selection
import pandas as pd
import numpy as np
######################## STILL IN WORK ###############################

class CustomSplit():

    def __init__(self, 
        train_val_test_size: tuple=(0.8,0.1,0.1),
        split_type: str="random",
        random_state: int = None
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

        self.split_type = split_type
        self.implemented_splits = ['random', 'k-fold', 'stratified']
        if self.split_type not in self.implemented_splits:
            print('[TrainTestSplit] Split type not implemented. Choosing random split')
            raise NotImplementedError("[TrainTestSplit] Split type {self.split_type} not implemented")

    def __call__(self, df:pd.DataFrame) -> tuple:
        if self.split_type == 'k-fold':
            _test_n_splits = int(1/self.test_size)
            test_splitter = model_selection.KFold(n_splits=)
            self.splitter = model_selection.KFold()




        