def s2b(s):
    """
    Convert a string to a bool. argparse does not handle passing bools
    """
    if isinstance(s, bool): return s 
    assert s in ['True', 'False'], f'expected argument to be \'True\' or \'False\', but got {s=}'
    return s == 'True'

def check_args(lst, args, keep=[]):
    """
    Take a list of modules and check in args whether to keep them. 
    Also enables forcing to keep something even if they are not in args using keep argument
    Parameters:
        lst: 
            list of modules to check
        args:
            args containing information on whether to keep the module. 
            The decision is True/False and stored in args.{name of module}
        keep:
            list of modules to keep even if args does not say so
    Returns:
        filtered modules according to args
    """
    result = []
    for module in lst:
        name = module.__class__.__name__ 
        if not name in keep:
            if vars(args)[name]:
                result.append(module)
        else: result.append(module)
    return result