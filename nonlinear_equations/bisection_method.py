from utils import BaseSolver, timing, print_iterations
    

class Bisection(BaseSolver):
    def __init__(self, func, interval, num_iterations=100, atol=1e-6, collect=True):
        super().__init__(num_iterations, atol, collect)
        self.f = func
        self.interval = interval
        self.a, self.b = interval
        
    @timing
    def solve(self):
        fa, fb = self.f(self.a), self.f(self.b)

        assert fa * fb < 0, "f(a) * f(b) must be less than 0"
        
        while abs(self.b - self.a) >= 2 * self.atol or self.i < self.n:
            r = (self.a + self.b) / 2
            fr = self.f(r)
            
            if self.collect:
                self.data[self.iter] = (r, fr)

            if fr == 0:
                self.root = r
                return r
            
            if fr * fa < 0:
                self.b = r
                fb = fr
            else:
                self.a = r
                fa = fr
            
            self.iter += 1
        
        self.root = r
        return r
    

def main():

    def func(x):
        return x**3 - 4*x + 1

    interval = [0, 1]
    num_iterations = 200
    atol = 1e-6
    
    solver = Bisection(func, interval, num_iterations, atol, collect=True)
    root = solver.solve()
    
    solver.print_result()
    print_iterations(solver.get_iteration_data()) 


if __name__ == '__main__':
    main()