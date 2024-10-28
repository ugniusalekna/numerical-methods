from utils import BaseSolver, timing, print_iterations


class Secant(BaseSolver):
    def __init__(self, func, x0, x1, num_iterations=100, atol=1e-6, collect=True):
        super().__init__(num_iterations, atol, collect)
        self.f = func
        self.x0 = x0
        self.x1 = x1
        
    @timing
    def solve(self):
                
        while self.iter < self.n:
            fx0 = self.f(self.x0)
            fx1 = self.f(self.x1)
                        
            r = self.x1 - fx1 * (self.x1 - self.x0) / (fx1 - fx0)
            
            if self.collect:
                self.data[self.iter] = (self.x1, r)
            
            if abs(r - self.x1) < self.atol:
                self.root = r
                return r

            self.x0 = self.x1
            self.x1 = r
            self.iter += 1
            
        self.root = r
        return r
    

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
    
    solver.print_result()
    print_iterations(solver.get_iteration_data())
    

if __name__ == '__main__':
    main()