import numpy as np

from utils import DirectSolver


class GaussianElimination(DirectSolver):
    def __init__(self, A, b):
        super().__init__(rhs=b)
        self.A = A
        self.b = b
                
    def forward_elimination(self):
        for j in range(self.n):
            if self.A[j, j] == 0:
                for k in range(j+1, self.n):
                    if self.A[k, j] != 0:
                        self.A[[j, k]] = self.A[[k, j]]
                        self.b[[j, k]] = self.b[[k, j]]
                        break
                        
            pivot = self.A[j, j]
            
            for k in range(j+1, self.n):
                factor = self.A[k, j] / pivot
                self.A[k, j:] -= self.A[j, j:] * factor
                self.b[k] -= self.b[j] * factor

    def backward_substitution(self):        
        for j in range(self.n-1, -1, -1):
            self.x[j] = (self.b[j] - np.dot(self.A[j, j+1:], self.x[j+1:])) / self.A[j, j]
            
        return self.x


def main():
    A = np.array([
        [2, 1, -1], 
        [-3, -1, 2], 
        [-2, 1, 2]
    ], dtype=np.float32)
    
    b = np.array([8, -11, -3], dtype=np.float32)

    gauss = GaussianElimination(A, b)
    solution = gauss.solve()
    print("Solution:", solution)


if __name__ == '__main__':
    main()