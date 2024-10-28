import numpy as np

from utils import IterativeSolver, print_iterations, generate_matrix


class JacobiMethod(IterativeSolver):
    def _initialize(self):
        self.x = np.zeros(self.n)
        
    def _update(self):
        self.x_prev = self.x.copy()
        self.x = (self.b - np.dot(self.A - np.diag(np.diag(self.A)), self.x)) / np.diag(self.A)

    
def main():
    A, b = generate_matrix(size=1000, seed=42)

    solver = JacobiMethod(A, b, atol=1e-6, num_iterations=10000, collect=True)
    solution = solver.solve()
    
    print_iterations(solver.get_iteration_data(), m=5)


if __name__ == '__main__':
    main()