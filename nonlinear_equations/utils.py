import time
from functools import wraps
from abc import ABC, abstractmethod


def timing(f):
    @wraps(f)
    def wrapper(*a, **kw):
        ts = time.time()
        out = f(*a, **kw)
        te = time.time()
        t = te - ts
        print(f"Function '{f.__name__}' took {t:.6f} seconds to execute.")
        return out
    return wrapper

            
class SolverBase(ABC):
    def __init__(self, n=100, atol=1e-6, collect=True):
        self.n = n
        self.atol = atol
        self.collect = collect
        self.reset()
        
    def reset(self):
        self.data = {}
        self.iter = 0
        self.root = None
    
    @abstractmethod
    def _update(self):
        pass
    
    @abstractmethod
    def _stop_criterion(self):
        pass
    
    @abstractmethod
    def _collect(self):
        pass
    
    @timing
    def solve(self):        
        for self.iter in range(self.n):
            self._update()
            
            if self.collect:
                self.data[self.iter] = self._collect()

            if self._stop_criterion():
                print(f"\nConverged in {self.iter+1} iteration(s).")
                return self.root
            
            self.iter += 1

        print(f"\nMaximum iterations reached ({self.n}).")
        return self.root
    
    def get_iteration_data(self):
        assert self.collect, "Iteration data is not available unless collect=True."
        return self.data


def print_iterations(data, m=5):
    tot = len(data)
    
    keys = list(data[0].keys())
    header = f"{'iter':<10}"
    column_widths = {}
    
    def format_value(value, width):
        if isinstance(value, (int, float)):
            return f"{value:<{width}.8f}"
        return str(value).ljust(width)

    def print_row(i, row_data):
        row = f"{i + 1:<10}"
        for key in keys:
            if key not in column_widths:
                continue
            row += f" {format_value(row_data[key], column_widths[key])}"
        print(row)
    
    for key in keys:
        width = 20
        column_widths[key] = width
        header += f" {key:<{width}}"
    
    print(header)
    print('-' * len(header))
    
    if tot <= 2 * m:
        indices = range(tot)
    else:
        indices = list(range(m)) + ['...'] + list(range(tot - m, tot))

    for i in indices:
        if i == '...':
            row = f"{'...':<10}" + "".join(f" {'...'.ljust(column_widths[key])}" for key in column_widths)
            print(row)
        else:
            print_row(i, data[i])