from ablation import Ablator 
def main():
    kwargs = {
        'epochs': 5, 
        'N': 100, 
        'wandb': False, 
        'sr': 1, 
        'max_duration': 30000,
        'duration': 1500, 
        'batch_size': 2
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
    ablator(**kwargs)

     


if __name__ == '__main__':
    main()