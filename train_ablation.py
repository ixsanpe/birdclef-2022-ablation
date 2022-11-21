from ablation import Ablator 
import time 
def main():
    kwargs = {
        'epochs': 10, 
        'N': -1, 
        'wandb': True, 
        'project_name': 'AblationTest',
        'experiment_name': 'ablation_' + str(int(time.time())), 
        'sr': 1, 
        'max_duration': 3000,
        'duration': 500, 
        'batch_size_train': 16, 
        'batch_size_val': 1, 
        'validate_every': 150, 
        'precompute': 'True', 
        'n_splits': 5,
    }

    modules = [ 
        'InstanceNorm', 
    ]

    default_bool = False 
    ablator = Ablator(
        script_name='train_runnable.py', 
        default_bool=default_bool, 
        modules=modules
    )
    ablator(run_reference=True, **kwargs)

     


if __name__ == '__main__':
    main()