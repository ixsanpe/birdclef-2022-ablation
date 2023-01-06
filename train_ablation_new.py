from ablation import Ablator 
from datetime import datetime 
"""
This script runs an ablation study, using the new baseline.
To that end, it calls train.py with different parameters through the class Ablator.

Ablator runs ablation studies as follows:
if run_reference: run a reference run with default parameters
After this, it goes through each module in modules and includes/ecxludes it for one run.
Example: 
    modules = [module1, module2, module3], default_bool = False
    Then we make 3 runs:
    Run    module1     module2     module3
    1.     True        False       False
    2.     False       True        False
    3.     False       False       True
    Here, True indicates that the module is included, and False otherwise

Finally, it goes through the dict sweeping and tries the specified values.

Example:
    sweeping = {learning_rate: [2, 3], model_name: [vgg_net]}
    Run     learning_rate       model_name
    1.      2                   resnet (default)
    2.      3                   resnet (default)
    --- Switch to change model name since we have tried each learning_rate ---
    3.      1 (default)         vgg_net
"""
def main():
    kwargs = {
        'epochs': 10, 
        'N': -1, 
        'wandb': True, 
        'project_name': 'AblationTest',
        'experiment_name': 'ablation_' + datetime.now().strftime("%Y-%m-%d-%H-%M"),
        'sr': 1, 
        'max_duration': 1500,
        'duration': 1500, 
        'batch_size_train': 16, 
        'batch_size_val': 1, 
        'validate_every': -1, 
        'precompute': 'True', 
        'n_splits': 5,
        'test_split': .05,
        'model_name': 'resnet34', 
        'scheme': 'new', 
        'k_runs': 3, 
        'policy': 'rolling_avg', 
        'augs': 'shift'
    }

    modules = [ # modules to include or exclude (changed one at a time from the default boolean)
        'InstanceNorm', 
    ]

    sweeping = { # Specify the alternatives to the default. These are tried one by one. 

        'augs': ['timestretch', 'backgroundnoise', 'frequencymask', 'gain', 'gaussiannoise', 'pitchshift', 'timemask'],  
        'model_name': ['efficientnet_b2', 'eca_nfnet_l0'], 
        'learning_rate': [1e-2,  1e-4], 
        'policy': ['first_and_final', 'max_all'], 
        'duration': [500, 1000], 
        'loss': ['FocalLoss', 'WeightedBCELoss', 'WeightedFocalLoss']
    }

    default_bool = False # whether to include each module in modules by default

    ablator = Ablator(
        script_name='train.py', 
        default_bool=default_bool, 
        modules=modules, 
        sweeping=sweeping
    )
    ablator(run_reference=True, **kwargs)

     


if __name__ == '__main__':
    main()