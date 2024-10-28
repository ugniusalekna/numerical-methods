import numpy as np

from utils import DirectSolver


class ThomasAlgorithm(DirectSolver):
    def __init__(self, lower_diag, diag, upper_diag, rhs):
        super().__init__(rhs=rhs)
        self.a = lower_diag
        self.b = diag
        self.c = upper_diag
        self.d = rhs
        
    def forward_elimination(self):
        for i in range(1, self.n):
            w = self.a[i-1] / self.b[i-1]
            self.b[i] -= w * self.c[i-1]
            self.d[i] -= w * self.d[i-1]

    def backward_substitution(self):
        self.x[-1] = self.d[-1] / self.b[-1]
        
        for i in range(self.n-2, -1, -1):
            self.x[i] = (self.d[i] - self.c[i] * self.x[i+1]) / self.b[i]
            
        return self.x

    
def main():
    lower_diag = np.array([1, 1], dtype=np.float32)
    diag = np.array([4, 4, 4], dtype=np.float32)
    upper_diag = np.array([1, 1], dtype=np.float32)
    rhs = np.array([5, 5, 5], dtype=np.float32)

    solver = ThomasAlgorithm(lower_diag, diag, upper_diag, rhs)
    solution = solver.solve()
    print("Solution:", solution)


if __name__ == '__main__':
    main()