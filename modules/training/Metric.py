from typing import Callable

class Metric():
    def __init__(self, name: str, metric: Callable):
        """
        A class that can compute a metric and also maintains a name for that metric
        """
        super().__init__()
        self.name = name 
        self.metric = metric
    
    def __call__(self, x, y):
        return self.metric(x, y)
