import time
from functools import wraps


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


def print_iterations(iteration_data, m=5):
    tot = len(iteration_data)
    
    print(f"{'Iter.i':<10} {'x_i':<25} {'f(x_i)':<25}")
    print('-' * 60)
    
    if tot <= 2 * m:
        for i, (x_i, f_x_i) in iteration_data.items():
            print(f"{i+1:<10} {x_i:<25.14f} {f_x_i:<25.14f}")
    else:
        for i in range(m):
            x_i, f_x_i = iteration_data[i]
            print(f"{i+1:<10} {x_i:<25.14f} {f_x_i:<25.14f}")
        
        print(f"{'...':<10} {'...':<25} {'...':<25}")
        
        for i in range(tot - m, tot):
            x_i, f_x_i = iteration_data[i]
            print(f"{i+1:<10} {x_i:<25.14f} {f_x_i:<25.14f}")
            
            
class BaseSolver:
    def __init__(self, n=100, atol=1e-6, collect=True):
        self.n = n
        self.atol = atol
        self.collect = collect
        
        self.data = {}
        self.iter = 0
        self.root = None
        
    @timing
    def solve(self):
        raise NotImplementedError("Must implement this method.")

    def get_iteration_data(self):
        assert self.collect, "Iteration data is not available unless collect=True."
        return self.data
    
    def print_result(self):
        if self.root is None:
            print("Run the solve method first.")
        elif self.iter < self.n:
            print(f"Root found: {self.root:.10f} after {self.iter+1} iterations.")
        else:
            print(f"Maximum iterations ({self.n}) reached. Last approximation: {self.root:.10f}")