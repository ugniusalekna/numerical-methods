import numpy as np

from utils import IterativeSolver, print_iterations, generate_matrix
from plotting import plot_omega_vs_iterations


def spectral_radius(A, omega):
    I = np.eye(A.shape[0])
    
    D_inv = np.diag(1 / np.diag(A))
    L = -D_inv @ np.tril(A, -1)
    U = -D_inv @ np.triu(A, 1)
    
    M_inv = np.linalg.inv(I - omega * L)
    R = np.dot(M_inv, (1 - omega) * I + omega * U)

    eigvals = np.linalg.eigvals(R)
    
    return max(abs(eigvals))


def optimal_omega(A):
    I = np.eye(A.shape[0])
    
    D_inv = np.diag(1 / np.diag(A))
    C = I - np.dot(D_inv, A)
    
    eigvals = np.linalg.eigvals(C)
    omega = 2 / (1 + np.sqrt(1 - max(abs(eigvals))**2))
    
    print(f"Computed optimal omega: {omega:.6f}; spectral radius: {spectral_radius(A, omega):.6f}")
    return omega


class SORMethod(IterativeSolver):
    def __init__(self, A, b, omega=None, optimize_omega=False, **kwargs):
        super().__init__(A, b, **kwargs)
        self.omega = omega if omega is not None else (optimal_omega(A) if not optimize_omega else 1.0)
        
        self.optimize_omega = optimize_omega
        self.x_prev, self.x_prev_prev = None, None

    def get_spectral_radius(self, omega):
        return spectral_radius(self.A, omega)
    
    def approx_spectral_radius(self):
        if self.x_prev is None or self.x_prev_prev is None:
            return 1.0
        
        num = np.linalg.norm(self.x - self.x_prev)
        den = np.linalg.norm(self.x_prev - self.x_prev_prev)
        return num / den if den != 0 else 1.0

    def update_omega(self, rho):
        self.omega = 2 / (1 + np.sqrt(1 - rho**2)) if rho < 1 else 1.0

    def _initialize(self):
        self.x = np.zeros(self.n)
        
    def _update(self):
        if self.optimize_omega:
            self.x_prev_prev = self.x_prev.copy() if self.x_prev is not None else self.x.copy()
        
        self.x_prev = self.x.copy()    
        
        for i in range(self.n):
            sum_ = np.dot(self.A[i, :i], self.x[:i]) + np.dot(self.A[i, i+1:], self.x[i+1:])
            self.x[i] = (1 - self.omega) * self.x[i] + self.omega * (self.b[i] - sum_) / self.A[i, i]

        if self.optimize_omega:
            rho = self.approx_spectral_radius()
            self.update_omega(rho)


def main():
    # A, b = generate_matrix(size=1000, symmetric=True, seed=42)
    A = np.array([
        [1, -0.95, 0],
        [-0.95, 2, -0.95],
        [0, -0.95, 1]
    ])

    b = np.array([0.05, 0.1, 0.05])

    omega = None
    
    solver = SORMethod(A, b, omega=omega, atol=0.0001, num_iterations=10000, collect=True)
    solution = solver.solve()
    
    print_iterations(solver.get_iteration_data(), m=5, show_x=True)
    
    plot_omega_vs_iterations(solver, bounds=[0.1, 2.0], step=0.01, add_opt=True, plot_rho=True)


if __name__ == '__main__':
    main()