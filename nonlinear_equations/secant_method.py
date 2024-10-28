from utils import BaseSolver, timing


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
            
            if abs(r - self.x1) < self.atol:
                self.root = r
                return r
            
            if self.collect:
                self.data[self.iter] = (self.x1, r)

            self.x0 = self.x1
            self.x1 = r
            self.iter += 1
            
        self.root = r
        return r
    

def f(x):
    return x**2 - 2


def main():
    x0 = 1.0
    x1 = 1.01
    num_iterations = 200
    atol = 1e-6
    
    secant = Secant(f, x0, x1, num_iterations, atol, collect=True)
    root = secant.solve()
    
    secant.print_result()
    secant.print_iterations()
    

if __name__ == '__main__':
    main()