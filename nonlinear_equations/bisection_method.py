from utils import SolverBase, print_iterations
    

class BisectionMethod(SolverBase):
    def __init__(self, func, interval, num_iterations=100, atol=1e-6, collect=True):
        super().__init__(num_iterations, atol, collect)
        self.func = func
        self.a, self.b = interval
        
        self.fa = self.func(self.a)
        self.fb = self.func(self.b)
        
        assert self.fa * self.fb < 0, "f(a) * f(b) must be less than 0"

    def _stop_criterion(self):
        return abs(self.b - self.a) < 2 * self.atol or self.fr == 0
    
    def _update(self):
        self.root = (self.a + self.b) / 2
        self.fr = self.func(self.root)

        if self.fr * self.fa < 0:
            self.b = self.root
            self.fb = self.fr
        else:
            self.a = self.root
            self.fa = self.fr
    
    def _collect(self):
        return {'x': self.root}


def main():

    def func(x):
        return x**3 - 4*x + 1

    interval = [0, 1]
    num_iterations = 200
    atol = 1e-6
    
    solver = BisectionMethod(func, interval, num_iterations, atol, collect=True)
    root = solver.solve()
    
    print_iterations(solver.get_iteration_data()) 


if __name__ == '__main__':
    main()