from utils import SolverBase, print_iterations


class NewtonRaphson(SolverBase):
    def __init__(self, func, deriv, x0, num_iterations=100, atol=1e-6, collect=True):
        super().__init__(num_iterations, atol, collect)
        self.func = func
        self.dfunc = deriv 
        self.x0 = x0
        self.root = self.x0
        
    def _stop_criterion(self):
        return abs(self.root - self.root_prev) < self.atol
        
    def _update(self):
        self.root_prev = self.root

        self.fx = self.func(self.root)
        self.dfdx = self.dfunc(self.root)
        
        if self.dfdx == 0:
            raise ValueError("Derivative is zero. Newton's method fails.")
        
        self.root = self.root - self.fx / self.dfdx
    
    def _collect(self):
        return {'x': self.root}


def main():

    def f(x):
        return x**2 - 2

    def df(x):
        return 2 * x

    x0 = 1.0
    num_iterations = 200
    atol = 1e-6
    
    solver = NewtonRaphson(f, df, x0, num_iterations, atol, collect=True)
    root = solver.solve()
    
    print_iterations(solver.get_iteration_data()) 
    

if __name__ == '__main__':
    main()