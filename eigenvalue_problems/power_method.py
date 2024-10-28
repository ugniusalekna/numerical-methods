import numpy as np
from utils import IterativeBase, print_iterations


class PowerMethod(IterativeBase):
    def _initialize(self):
        self.x = np.random.rand(self.n)
        self.x /= np.linalg.norm(self.x)
        self.eig = 0
        
    def _update(self):
        self.x_prev = self.x.copy()

        self.y = np.dot(self.A, self.x)
        self.x = self.y / np.linalg.norm(self.y)
        
        self.eig = np.dot(self.y, self.x_prev)
        
    def _error(self):
        return np.linalg.norm(self.y - self.eig * self.x_prev, ord=2)
        
    def _stop_criterion(self, err):
        return err < self.atol

    def _collect(self, err):
        return {'eigenvector': self.x.copy(), 'max_eigenvalue': self.eig, 'error': err}
        
    def _return(self):
        return self.x, self.eig


class InversePowerMethod(PowerMethod):
    def _update(self):
        self.x_prev = self.x.copy()

        self.y = np.linalg.solve(self.A, self.x) # A.y=x <=> y=A^-1.x
        self.x = self.y / np.linalg.norm(self.y)
        
        self.eig = np.dot(self.y, self.x_prev)

    def _collect(self, err):
        return {'eigenvector': self.x.copy(), 'min_eigenvalue': 1 / self.eig, 'error': err}
        
    def _return(self):
        return self.x, 1 / self.eig


def find_nearest_eigenvalue(A, mu=0, atol=1e-8, num_iterations=10000):
    A_sh = A - mu * np.eye(A.shape[0])
    
    s = InversePowerMethod(A_sh, atol=atol, num_iterations=num_iterations, collect=True)
    _, eig_min = s.solve()
    
    eig_mu = eig_min + mu
        
    return eig_mu


def find_condition_number(A, atol=1e-8, num_iterations=10000):
    symmetric = np.all(A == A.T) 

    if symmetric:
        A_ = A.T @ A

    s = PowerMethod(A_, atol=atol, num_iterations=num_iterations)
    _, max_val = s.solve()
    
    s = InversePowerMethod(A_, atol=atol, num_iterations=num_iterations)
    _, min_val = s.solve()

    rho = abs(max_val / min_val)
    if not symmetric:
        rho = np.sqrt(rho)
        
    return rho
    


def main():
    A = np.array([
        [4, 1, 2, 0],
        [1, 3, 0, 1],
        [2, 0, 2, 1],
        [0, 1, 1, 3]
    ])
    
    solver = PowerMethod(A, atol=1e-8, num_iterations=10000)
    x, eig_max = solver.solve()
    print_iterations(data=solver.get_iteration_data(), m=5, show_vectors=True)
    
    solver = InversePowerMethod(A, atol=1e-8, num_iterations=10000)
    x, eig_min = solver.solve()
    print_iterations(data=solver.get_iteration_data(), m=5, show_vectors=True)
    
    print(f'\nEigenvalues (min-max) of A: {eig_min:.6f}, {eig_max:.6f}')
    print(f'Spectral radius of A: {abs(eig_max):.6f}.')
    
    rho = find_condition_number(A)
    print(f'Spectral condition number of A: {rho:.6f}.')

    mu = 3.5
    eig_mu = find_nearest_eigenvalue(A, mu=mu)
    print(f'Eigenvalue closest to {mu:.2f}: {eig_mu:.6f}.')


if __name__ == '__main__':
    main()