import argparse
from ablation_utils import s2b

N = 5


def main():
    parser = argparse.ArgumentParser(description='This is your Argparser')

    parser.add_argument('--default_bool', type=s2b, default='True')
    parser.add_argument('--wandb', type=s2b, default='True')
    parser.add_argument('--wandb_name', type=str, default='my_default_wandb')
    parser.add_argument('--lr', type=float)
    args = parser.parse_known_args() 

    default = args[0].default_bool
    for i in range(N):
        parser.add_argument(f'--choice{i}', type=s2b, default=default)
    args = parser.parse_args()

    print(f'{args=}')
    

if __name__ == '__main__':
    main()
