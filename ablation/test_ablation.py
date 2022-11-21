from Ablator import Ablator
from argparse_experiments import N

def main():
    """
    Run a mock ablation study setting a default value for each choice and 
    changing it for one module at a time
    """
    script_name = 'ablation/argparse_experiments.py'
    modules = [f'choice{i}' for i in range(N - 2)]
    default_bool = False
    kwargs = {'wandb': False, 'choice0':True, 'lr': 1e-4}
    ablator = Ablator(script_name=script_name, default_bool=default_bool, modules=modules)
    ablator(**kwargs)

if __name__ == '__main__':
    main()