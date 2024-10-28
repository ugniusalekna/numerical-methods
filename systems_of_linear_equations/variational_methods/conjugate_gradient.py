import numpy as np

from utils import VariationalSolver, print_iterations, generate_matrix


class ConjugateGradient(VariationalSolver):
    def _initialize(self):
        self.x = np.zeros(self.n)
        self.r = self.b - np.dot(self.A, self.x)
        self.p = self.r.copy()
        self.r_dot_prev = np.dot(self.r, self.r)

    def _update(self):
        Ap = np.dot(self.A, self.p)
        tau = self.r_dot_prev / np.dot(self.p, Ap)
        
        self.x += tau * self.p
        self.r -= tau * Ap

        r_dot = np.dot(self.r, self.r)
        beta = r_dot / self.r_dot_prev

        self.p = self.r + beta * self.p
        self.r_dot_prev = r_dot


def main():
    # A, b = generate_matrix(size=1000, symmetric=True, seed=42)
    A = np.array([
        [2, 1, 0.95],
        [1, 2, 1],
        [0.95, 1, 2]
    ])

    b = np.array([3.95, 4, 3.95])

    solver = ConjugateGradient(A, b, atol=0.0001, num_iterations=10000, collect=True)
    solution = solver.solve()

    print_iterations(solver.get_iteration_data(), m=5, show_x=True)


if __name__ == '__main__':
    main()