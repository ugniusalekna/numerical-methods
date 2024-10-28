from utils import SolverBase, print_iterations


class Secant(SolverBase):
    def __init__(self, func, x0, x1, num_iterations=100, atol=1e-6, collect=True):
        super().__init__(num_iterations, atol, collect)
        self.func = func
        self.x0 = x0
        self.x1 = x1
        
        self.fx0 = self.func(self.x0)
        self.fx1 = self.func(self.x1)
        
    def _stop_criterion(self):
        return abs(self.x0 - self.x1) < self.atol
        
    def _update(self):
        self.root = self.x1 - self.fx1 * (self.x1 - self.x0) / (self.fx1 - self.fx0)
        
        self.x0, self.fx0 = self.x1, self.fx1
        self.x1, self.fx1 = self.root, self.func(self.root)

    def _collect(self):
        return {'x': self.root}


def main():

    def f(x):
        return x**2 - 2

    h = 1e-6
    x0 = 1.0
    x1 = x0 + h
    
    num_iterations = 200
    atol = 1e-6
    
    solver = Secant(f, x0, x1, num_iterations, atol, collect=True)
    root = solver.solve()
    
    print_iterations(solver.get_iteration_data())
    

if __name__ == '__main__':
    main()