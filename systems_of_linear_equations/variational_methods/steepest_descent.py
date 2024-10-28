import numpy as np

from utils import VariationalSolver, print_iterations, generate_matrix


class SteepestDescent(VariationalSolver):
    def _initialize(self):
        self.x = np.zeros(self.n)
        self.r = self.b - np.dot(self.A, self.x)
        
    def _update(self):
        tau = np.dot(self.r, self.r) / np.dot(self.r, np.dot(self.A, self.r))
        self.x += tau * self.r
        
        self.r = self.b - np.dot(self.A, self.x)
        

def main():
    A, b = generate_matrix(size=1000, symmetric=True, seed=42)

    solver = SteepestDescent(A, b, atol=1e-6, num_iterations=10000, collect=True)
    solution = solver.solve()

    print_iterations(solver.get_iteration_data(), m=5)


if __name__ == '__main__':
    main()