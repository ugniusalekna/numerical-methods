import numpy as np

from utils import SolverBase, print_iterations


class FixedPointIteration(SolverBase):
    def __init__(self, func, x0, num_iterations=100, atol=1e-6, collect=True):
        super().__init__(num_iterations, atol, collect)
        self.func = func
        self.x0 = x0
        self.root = self.x0
    
    def _stop_criterion(self):
        return abs(self.root - self.root_prev) < self.atol
        
    def _update(self):
        self.root_prev = self.root
        self.root = self.func(self.root)
        
    def _collect(self):
        return {'x': self.root}


def main():

    def g(x):
        return np.sqrt(4 + x)

    x0 = 0
    num_iterations = 200
    atol = 1e-6
    
    solver = FixedPointIteration(g, x0, num_iterations, atol, collect=True)
    root = solver.solve()
    
    print_iterations(solver.get_iteration_data())


if __name__ == '__main__':
    main()