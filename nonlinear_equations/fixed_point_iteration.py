import numpy as np

from utils import BaseSolver, timing, print_iterations


class FixedPointIteration(BaseSolver):
    def __init__(self, func, x0, num_iterations=100, atol=1e-6, collect=True):
        super().__init__(num_iterations, atol, collect)
        self.f = func
        self.x0 = x0

    @timing
    def solve(self):
        x = self.x0
        
        while self.iter < self.n:
            r = self.f(x)
            
            if self.collect:
                self.data[self.iter] = (x, r)
            
            if abs(r - x) < self.atol:
                self.root = r
                return r
     
            x = r
            self.iter += 1
            
        self.root = r
        return r
    

def main():

    def g(x):
        return np.sqrt(4 + x)

    x0 = 0
    num_iterations = 200
    atol = 1e-6
    
    solver = FixedPointIteration(g, x0, num_iterations, atol, collect=True)
    root = solver.solve()
    
    solver.print_result()
    print_iterations(solver.get_iteration_data()) 


if __name__ == '__main__':
    main()