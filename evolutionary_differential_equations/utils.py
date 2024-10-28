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


class EvolutionaryBase(ABC):
    def __init__(self, func, y0, t0=0, h=0.01, collect=True, true_solution=None):
        self.func = func
        self.h = h
        self.t0 = t0
        self.y0 = y0
        self.collect = collect
        self.true_sol = true_solution
        self.reset()
        
    def reset(self):
        self.data = {}
        self.i = 0
        self.t_values = [self.t0]
        self.y_values = [self.y0]

    def _update(self, y, t):
        self.y_values.append(y)
        self.t_values.append(t)

    def _collect(self, t, y):
        entry = {'t': t.copy(), 'y': y.copy()}
        
        if self.true_sol:
            y_true = self.true_sol(t)
            err = np.linalg.norm(y - y_true, ord=2)
            entry.update({'y_true': y_true, 'error': err})
            
        return entry

    @abstractmethod
    def _step(self, y, t, h):
        pass
    
    @timing
    def integrate(self, tf):
        y = self.y0.copy()
        t = self.t0.copy()
        
        if self.collect:
            self.data[0] = self._collect(t, y)
            self.i = 1

        while t < tf:
            h = min(self.h, tf - t)
            y = self._step(y, t, h)
            t = t + h
            self._update(y, t)
            
            if self.collect:
                self.data[self.i] = self._collect(t, y)
                self.i += 1
        
        return np.array(self.y_values), np.array(self.t_values)
    
    def get_iteration_data(self):
        assert self.collect, "Iteration data not available unless collect=True."
        return self.data
    

def print_iteration_table(data, m=5, show_vectors=True):
    tot = len(data)
    
    keys = list(data[0].keys())
    column_widths = {}
    
    def format_value(value, width):
        if isinstance(value, (list, np.ndarray)) and show_vectors:
            return f"[{', '.join([f'{v:.5f}' for v in value])}]".ljust(width)
        elif isinstance(value, (int, float)):
            return f"{value:<{width}.8e}"
        return str(value).ljust(width)

    def print_row(i, row_data):
        row = f"{i:<10}"
        for key in keys:
            if key not in column_widths:
                continue
            row += f" {format_value(row_data[key], column_widths[key])}"
        print(row)
    
    header = f"{'step':<10}"
    for key in keys:
        if isinstance(data[0][key], (list, np.ndarray)) and show_vectors:
            width = len(format_value(data[0][key], width=0)) + 10
        elif isinstance(data[0][key], (list, np.ndarray)) and not show_vectors:
            continue
        else:
            width = 15
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