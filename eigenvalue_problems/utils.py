import time
import numpy as np
from functools import wraps
from abc import ABC, abstractmethod


def timing(f):
    @wraps(f)
    def wrapper(*a, **kw):
        ts = time.time()
        out = f(*a, **kw)
        te = time.time()
        t = te - ts
        print(f"Runtime of '{f.__name__}': {t:.6f} seconds.")
        return out
    return wrapper


class IterativeBase(ABC):
    def __init__(self, A, atol=1e-6, num_iterations=100, collect=True):
        self.A = A
        self.atol = atol
        self.num_iterations = num_iterations
        self.collect = collect
        self.reset()
        
    def reset(self):
        self.data = {}
        self.iter = 0
        self.x = None
        self.eig = None
        self.n = self.A.shape[0]
        self._initialize()

    @abstractmethod
    def _initialize(self):
        pass
    
    @abstractmethod
    def _update(self):
        pass

    @abstractmethod
    def _error(self):
        pass
    
    @abstractmethod
    def _stop_criterion(self, err):
        pass

    @abstractmethod
    def _collect(self, err):
        pass
    
    @abstractmethod
    def _return(self):
        pass
    
    @timing
    def solve(self):
        for self.iter in range(self.num_iterations):
            self._update()
            
            err = self._error()
            
            if self.collect:
                self.data[self.iter] = self._collect(err)
            
            if self._stop_criterion(err):
                print(f"\nConverged in {self.iter + 1} iteration(s).")
                return self._return()
        
        print(f"\nMaximum iterations reached ({self.num_iterations}).")
        return self._return()
    
    def get_iteration_data(self):
        assert self.collect, "Iteration data not available unless collect=True."
        return self.data


def print_iterations(data, m=5, show_vectors=False):
    tot = len(data)
    
    keys = list(data[0].keys())
    header = f"{'Iter':<10}"
    column_widths = {}
    
    def format_value(value, width):
        if isinstance(value, (list, np.ndarray)) and show_vectors:
            return f"[{', '.join([f'{v:.5f}' for v in value])}]".ljust(width)
        elif isinstance(value, (int, float)):
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
        if isinstance(data[0][key], (list, np.ndarray)) and show_vectors:
            width = len(format_value(data[0][key], width=0)) + 10
        elif isinstance(data[0][key], (list, np.ndarray)) and not show_vectors:
            continue
        else:
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