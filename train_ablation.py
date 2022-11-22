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
        'max_duration': 500,
        'duration': 500, 
        'batch_size_train': 16, 
        'batch_size_val': 1, 
        'validate_every': 150, 
        'precompute': 'True', 
        'n_splits': 5,
        'test_split': .05,
    }

    modules = [ # modules to include or exclude (changed one at a time from the default boolean)
        'InstanceNorm', 
    ]
    # TODO: add alternatives below!
    sweeping = { # first argument is the default, then it makes a run for each alternative
        'model_name': ['efficientnet_b2', 'resnet34', ], 
        # 'learning_rate': [1e-3, 1e-2, 1e-4] # Just as an example, we could have done this too
    }
    default_bool = False # whether to include each module in modules by default
    """
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
    
    Finally, it goes through sweep and tries the specified values, except the first, 
    which is the default value. This value we have already tried!
    Example:
        sweep = {learning_rate: [1, 2, 3], model_name: [resnet, vgg_net]}
        Run     learning_rate       model_name
        1.      2                   resnet (default)
        2.      3                   resnet (default)
        --- Switch to change model name since we have tried each learning_rate ---
        3.      1 (default)         vgg_net
    """
    ablator = Ablator(
        script_name='train_runnable.py', 
        default_bool=default_bool, 
        modules=modules, 
        sweeping=sweeping
    )
    ablator(run_reference=True, **kwargs)

     


if __name__ == '__main__':
    main()