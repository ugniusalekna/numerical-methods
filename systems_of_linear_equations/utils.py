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
        print(f"Function '{f.__name__}' took {t:.6f} seconds to execute.")
        return out
    return wrapper


class DirectSolver(ABC):
    def __init__(self, rhs):
        self.n = len(rhs)
        self.x = np.zeros(self.n)
    
    @abstractmethod
    def forward_elimination(self):
        pass

    @abstractmethod
    def backward_substitution(self):
        pass
    
    @timing
    def solve(self):
        self.forward_elimination()
        return self.backward_substitution()
    

    
class IterativeBase(ABC):
    def __init__(self, A, b, num_iterations=100, atol=1e-6, collect=True):
        self.A = A
        self.b = b
        self.num_iterations = num_iterations
        self.atol = atol
        self.collect = collect
        self.reset()
        
    def reset(self):
        self.data = {}
        self.iter = 0
        self.x = None
        self.x_prev = None
        self.n = len(self.b)
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
    
    @timing
    def solve(self):
        for self.iter in range(self.num_iterations):
            self._update()
            
            err = self._error()
            
            if self.collect:
                self.data[self.iter] = self._collect(err)

            if self._stop_criterion(err):
                print(f"Converged in {self.iter + 1} iterations.")
                return self.x
        
        print(f"Maximum iterations reached ({self.num_iterations}).")
        return self.x
    
    def get_iteration_data(self):
        assert self.collect, "Iteration data not available unless collect=True."
        return self.data


class IterativeSolver(IterativeBase):
    def _collect(self, err):
        return {'x': self.x.copy(), 'error': err}

    def _error(self):
        x_norm = np.linalg.norm(self.x - self.x_prev, ord=np.inf)
        err_norm = np.linalg.norm(self.b - np.dot(self.A, self.x), ord=np.inf)
        return x_norm + err_norm

    def _stop_criterion(self, err):
        return err < self.atol


class VariationalSolver(IterativeBase):
    def _collect(self, err):
        return {'x': self.x.copy(), 'error': err}
    
    def _error(self):
        return np.linalg.norm(self.r, ord=2)

    def _stop_criterion(self, err):
        return err < self.atol
    

def generate_matrix(size, symmetric=False, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    A = np.random.rand(size, size) * 10
    
    for i in range(size):
        A[i, i] = np.sum(np.abs(A[i])) + np.random.uniform(1, 10)
    
    if symmetric:
        A = (A + A.T) / 2
    
    b = np.random.rand(size) * 10
    
    return A, b


def print_iterations(data, m=5, show_vectors=False):
    tot = len(data)
    
    keys = list(data[0].keys())
    header = f"{'iter':<10}"
    column_widths = {}
    
    def format_value(value, width):
        if isinstance(value, (list, np.ndarray)) and show_vectors:
            return f"[{', '.join([f'{v:.5f}' for v in value])}]".ljust(width)
        elif isinstance(value, (int, float)):
            return f"{value:<{width}.8e}"
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