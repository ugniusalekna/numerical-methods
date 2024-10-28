from utils import BaseSolver, timing, print_iterations


class NewtonRaphson(BaseSolver):
    def __init__(self, func, deriv, x0, num_iterations=100, atol=1e-6, collect=True):
        super().__init__(num_iterations, atol, collect)
        self.f = func
        self.df = deriv 
        self.x0 = x0
        
    @timing
    def solve(self):
        x = self.x0
        
        while self.iter < self.n:
            fx = self.f(x)
            dfdx = self.df(x)
            
            if dfdx == 0:
                raise ValueError("Derivative is zero. Newton's method fails.")
            
            r = x - fx / dfdx
            
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

    def f(x):
        return x**2 - 2

    def df(x):
        return 2 * x

    x0 = 1.0
    num_iterations = 200
    atol = 1e-6
    
    solver = NewtonRaphson(f, df, x0, num_iterations, atol, collect=True)
    root = solver.solve()
    
    solver.print_result()
    print_iterations(solver.get_iteration_data()) 
    

if __name__ == '__main__':
    main()