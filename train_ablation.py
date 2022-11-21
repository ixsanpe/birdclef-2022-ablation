from ablation import Ablator 
import time 
"""
This script runs an ablation study.
To that end, it calls train_runnable.py with different parameters.
Currently, we only ablate on including different modules (in fact, 
only InstanceNorm), but it would be easy to extend the functionality
to try out different learning rates, whether to precompute 
spectrograms, including/excluding other modules, etc. 
"""
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