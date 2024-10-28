import numpy as np


class CholeskyDecomposition:
    def __init__(self, A, b):
        self.A = A
        self.b = b
        
        self.L = np.zeros_like(A)
        self.n = len(b)
        
    def decompose(self):
        for i in range(self.n):
            for j in range(i+1):
                sum_ = np.dot(self.L[i, :j], self.L[j, :j])
                
                if i == j:
                    self.L[i, j] = np.sqrt(self.A[i, j] - sum_)
                    
                else:
                    self.L[i, j] = (self.A[i, j] - sum_) / self.L[j, j]

    def forward_substitution(self):
        # solve Ld = b, d = L.Tx
        d = np.zeros(self.n)
        
        for i in range(self.n):
            d[i] = (self.b[i] - np.dot(self.L[i, :i], d[:i])) / self.L[i, i]

        return d
        
    def backward_substitution(self, d):
        # solve L.Tx = d
        x = np.zeros(self.n)
        
        for i in range(self.n-1, -1, -1):
            x[i] = (d[i] - np.dot(self.L.T[i, i+1:], x[i+1:])) / self.L.T[i, i]
        
        return x

    def solve(self):
        self.decompose()
        d = self.forward_substitution()
        return self.backward_substitution(d)
    

def main():
    A = np.array([
        [4, 12, -16],
        [12, 37, -43],
        [-16, -43, 98]
    ], dtype=np.float32)

    b = np.array([1, 2, 3], dtype=np.float32)

    solver = CholeskyDecomposition(A, b)
    solution = solver.solve()
    print("Solution:", solution)


if __name__ == '__main__':
    main()