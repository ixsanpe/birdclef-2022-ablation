import subprocess 

class AblationRun():
    def __init__(self, 
        script_name: str, 
        default_bool: bool, 
        modules: list[str], 
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
        self.modules = modules
    
    def __call__(self, **kwargs):
        # assign values to args
        args = kwargs
        # args['default_bool'] = self.default_bool
        for module in self.modules: 
            if module in self.exceptions:
                args[module] = not self.default_bool
            else:
                args[module] = self.default_bool
        
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
    def __init__(self, script_name: str, default_bool: bool, modules: list[str], sweeping: dict):
        """
        Run an ablation study for script_name by including/excluding modules
        Parameters:
            script_name:
                name of script to run a study on
            default_bool:
                The default choice to include modules
            modules:
                Candidate modules to exclude
            sweeping:
                dict with keys equal to which feature to change to each of the alternatives 
                in the dict's values. Its values are lists. 
        """
        self.script_name = script_name
        self.default_bool = default_bool 
        self.modules = modules 
        self.sweeping = sweeping
    
    def __call__(self, run_reference: bool=True, **kwargs):
        """
        For each module, run a script where this module is not set to 
        the value self.default_bool
        """
        # Do a run where everything is regular
        if run_reference:
            run = AblationRun(self.script_name, self.default_bool, modules=self.modules, exceptions=[])
            run(**kwargs)
        # include/exclude each module
        for module in self.modules:
            # print(f'\n\nChanging {module=}\n\n')
            run = AblationRun(self.script_name, self.default_bool, modules=self.modules, exceptions=[module])
            alternative_kwargs = kwargs.copy()
            alternative_kwargs['experiment_name'] = f'{module}_{not self.default_bool}_{kwargs["experiment_name"]}'
            run(**alternative_kwargs)
        # Change other alternatives like model_name/learning rate
        for sweep, alternatives in self.sweeping.items():
            for alternative in alternatives: 
                # print(f'\n\nChanging {sweep=}\n\n')
                run = AblationRun(self.script_name, self.default_bool, modules=self.modules, exceptions=[])
                alternative_kwargs = kwargs.copy()
                alternative_kwargs[sweep] = alternative
                alternative_kwargs['experiment_name'] = f'{sweep}_{alternative}_{kwargs["experiment_name"]}'
                run(**alternative_kwargs)
        
