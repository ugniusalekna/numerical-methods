import time
import numpy as np
from functools import wraps
from abc import ABC, abstractmethod


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


def print_iterations(iteration_data, m=5, show_x=False):
    tot = len(iteration_data)
    
    if show_x:
        print(f"{'Iter':<10} {'Residual':<25} {'X'}")
        print('-' * 100)
    else:
        print(f"{'Iter':<10} {'Residual':<25}")
        print('-' * 35)
    
    def format_x(x):
        return "[" + ", ".join([f"{elem:.5f}" for elem in x]) + "]"
    
    if tot <= 2 * m:
        for i, (x, res) in iteration_data.items():
            if show_x:
                x_str = format_x(x)
                print(f"{i+1:<10} {res:<25.14f} {x_str}")
            else:
                print(f"{i+1:<10} {res:<25.14f}")
    else:
        for i in range(m):
            x, res = iteration_data[i]
            if show_x:
                x_str = format_x(x)
                print(f"{i+1:<10} {res:<25.14f} {x_str}")
            else:
                print(f"{i+1:<10} {res:<25.14f}")
        
        if show_x:
            print(f"{'...':<10} {'...':<25} {'...':<70}")
        else:
            print(f"{'...':<10} {'...':<25}")
        
        for i in range(tot - m, tot):
            x, res = iteration_data[i]
            if show_x:
                x_str = format_x(x)
                print(f"{i+1:<10} {res:<25.14f} {x_str}")
            else:
                print(f"{i+1:<10} {res:<25.14f}")


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
    def _stop_criterion(self):
        pass

    @timing
    def solve(self):
        for self.iter in range(self.num_iterations):
            self._update()
            
            z = self._error()
            
            if self.collect:
                self.data[self.iter] = (self.x.copy(), z.copy())
            
            if self._stop_criterion(z):
                print(f"Converged in {self.iter + 1} iterations.")
                return self.x
        
        print(f"Maximum iterations reached ({self.num_iterations}).")
        return self.x
    
    def get_iteration_data(self):
        assert self.collect, "Iteration data not available unless collect=True."
        return self.data


class IterativeSolver(IterativeBase):
    def _error(self):
        x_norm = np.linalg.norm(self.x - self.x_prev, ord=np.inf)
        err_norm = np.linalg.norm(self.b - np.dot(self.A, self.x), ord=np.inf)
        return x_norm + err_norm

    def _stop_criterion(self, z):
        return z < self.atol


class VariationalSolver(IterativeBase):
    def _error(self):
        return np.linalg.norm(self.r, ord=2)

    def _stop_criterion(self, z):
        return z < self.atol