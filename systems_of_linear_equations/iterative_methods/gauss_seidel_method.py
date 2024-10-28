import numpy as np

from utils import IterativeSolver, print_iterations, generate_matrix


class GaussSeidelMethod(IterativeSolver):
    def _initialize(self):
        self.x = np.zeros(self.n)

    def _update(self):
        self.x_prev = self.x.copy()

        for i in range(self.n):
            sum_ = np.dot(self.A[i, :i], self.x[:i]) + np.dot(self.A[i, i+1:], self.x[i+1:])
            self.x[i] = (self.b[i] - sum_) / self.A[i, i]


def main():
    A, b = generate_matrix(size=1000, seed=42)

    solver = GaussSeidelMethod(A, b, atol=1e-6, num_iterations=10000, collect=True)
    solution = solver.solve()
        
    print_iterations(solver.get_iteration_data(), m=5)


if __name__ == '__main__':
    main()