import subprocess 

class AblationRun():
    def __init__(self, 
        script_name: str, 
        default_bool: bool, 
        exceptions: list[str], 
    ):
        """
        Create a single ablation run
        Parameters:
            script_name:
                name of the script to run
            default_bool:
                The default value to pass to boolean arguments
            exceptions:
                Arguments for which to pass not default_bool. 
        """
        self.default_bool = default_bool
        self.exceptions = exceptions 
        self.script_name = script_name
    
    def __call__(self, **kwargs):
        # assign values to args
        args = kwargs
        args['default_bool'] = self.default_bool
        for exception in self.exceptions:
            args[exception] = not self.default_bool
        
        # hack to make parsing work
        final_args = {}
        for k in args.keys():
            final_args['--' + k] = str(args[k]) 
        args = final_args
            
        # convert to list
        args = [[k, v] for k,v in args.items()]
        args = [i for sublist in args for i in sublist] # flatten list
        args = ['python', self.script_name] + args

        # run the program
        subprocess.call(args)

class Ablator():
    def __init__(self, script_name: str, default_bool: bool, modules: list[str]):
        """
        Run an ablation study for script_name by including/excluding modules
        Parameters:
            script_name:
                name of script to run a study on
            default_bool:
                The default choice to include modules
            modules:
                Candidate modules to exclude
        """
        self.script_name = script_name
        self.default_bool = default_bool 
        self.modules = modules 
    
    def __call__(self, run_reference: bool=True, **kwargs):
        """
        For each module, run a script where this module is not set to 
        the value self.default_bool
        """
        if run_reference:
            run = AblationRun(self.script_name, self.default_bool, [])
            run(**kwargs)
        for module in self.modules:
            run = AblationRun(self.script_name, self.default_bool, [module])
            run(**kwargs)
        
