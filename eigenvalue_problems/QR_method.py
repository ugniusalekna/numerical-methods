import numpy as np
from scipy.linalg import hessenberg

from utils import IterativeBase, print_iterations
from plotting import plot_gershgorin_circles


def wilkinson_shift(A):
    """Compute Wilkinson's shift based on bottom-right 2x2 submatrix."""
    d = (A[-2, -2] - A[-1, -1]) / 2
    mu = A[-1, -1] - (np.sign(d) * A[-2, -1]**2) / (abs(d) + np.sqrt(d**2 + A[-2, -1]**2))
    return mu


class QRMethod(IterativeBase):
    def _initialize(self):
        self.A = hessenberg(self.A)

    def _update(self):
        shift = wilkinson_shift(self.A)
        
        I = np.eye(self.n)
        Q, R = np.linalg.qr(self.A - shift * I)
        self.A = R @ Q + shift * I
        
        self.eig = np.diag(self.A)
    
    def _error(self):
        L = np.tril(self.A, -1)
        return np.linalg.norm(L, ord=np.inf)
    
    def _stop_criterion(self, err):
        return err < self.atol

    def _collect(self, err):
        return {'eigenvalues': self.eig.copy(), 'error': err}
    
    def _return(self):
        return self.eig
    
    
def lesp_matrix(n):
    A = np.tril(np.ones(n))
    for i in range(n):
        A[i, i] = -i - 1
    return A

def smoke_matrix(n):
    A = np.ones((n, n), dtype=complex)
    
    for i in range(n):
        phi = 2 * np.pi * i / n
        A[i, i] = np.cos(phi) + 1j * np.sin(phi)
    
    return A


def main():
    # A = lesp_matrix(n=5)
    A = smoke_matrix(n=7)

    solver = QRMethod(A, atol=1e-6, num_iterations=1000)
    eigvals = solver.solve()
    print_iterations(solver.get_iteration_data(), show_vectors=True)
    
    plot_gershgorin_circles(A, eigvals=eigvals)

    
if __name__ == '__main__':
    main()