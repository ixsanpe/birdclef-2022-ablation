import pandas as pd 
from torch import nn
import json 
from modules import WandbLogger


class DummyTrainer():
    def __init__(self, ) -> None:
        self.model = nn.Identity()

def main():
    run = 'reference'
    metrics = [
        'train_F1Ours', 'val_F1Ours', 'train_F1Micro', 
        'val_F1Micro', 'train_loss', 'val_loss', 
        'train_RecallOurs', 'val_RecallOurs', 
        'train_PrecisionOurs', 'val_PrecisionOurs'
    ]
    dfs = {}
    for metric in metrics:
        df = pd.read_csv(f'birdclef-2022/wandb_runs/{run}/{metric}.csv')
        df = df[[c for c in df.columns if not 'MIN' in c and not 'MAX' in c]]
        dfs[metric] = df
    
    with open(f'birdclef-2022/wandb_runs/{run}/config.json') as f:
        config = json.load(f)
    
    config = {k: v['value'] for k, v in config.items()}
    
    epochs = config['epochs']

    name = df.columns[-1].split(' ')[0]
    for k in range(3):
        logger = WandbLogger(DummyTrainer(), experiment_name=name + f'_{k=}', project_name='AblationTest', config=config)
        for epoch in range(epochs):
            stats = {m: df.iloc[epoch + k*epochs, -1] for m, df in dfs.items()}
            logger(stats, log_rankings=False)
        logger.finish_run()



if __name__ == '__main__':
    main()