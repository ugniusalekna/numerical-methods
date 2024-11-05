import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
np.set_printoptions(linewidth=150)

from plotting import show_plot
from utils import create_matrix


class Poisson1DBase(ABC):
    def __init__(self, N, f, domain=(0, 1)):
        self.N = N
        self.f = f
        self.a, self.b = domain
        
        self.h = (self.b - self.a) / (N + 1)
        self.x = np.linspace(self.a, self.b, N + 2)
        self.u = np.zeros(self.N+2)
        
        self.T = None

    @abstractmethod
    def create_matrix(self):
        pass
    
    @abstractmethod
    def solve(self):
        pass

    def plot_solution(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        show_plot(
            ax=ax,
            x_data=self.x,
            y_data=self.u,
            labels=[rf'$u_{{i}}, i = 1, ..., {{{self.N+2}}}$'],
            title="Solution of 1D Poisson equation",
            x_label=r"$x_{i}$",
            y_label=r"$u(x_{i}) = u_{i}$",
            markers=['o'],
            line_styles=['-'],
            colors=['b'],
            legend_loc="best",
            grid=True
        )
        plt.show()


class DirichletPoisson1D(Poisson1DBase):
    def __init__(self, N, f, u_a, u_b, domain=(0, 1)):
        super().__init__(N, f, domain)
        self.u_a = u_a
        self.u_b = u_b
        self.create_matrix()

    def create_matrix(self):
        self.T = create_matrix(N=self.N, alpha=1.0, beta=0.0, order=2)

    def solve(self):
        f_vec = self.f(self.x[1:-1]) * self.h**2
        f_vec[0] += self.u_a
        f_vec[-1] += self.u_b
        
        u_int = la.solve(self.T, f_vec)
        
        self.u[1:-1] = u_int
        self.u[0] = self.u_a
        self.u[-1] = self.u_b
        
        return self.u


class RobinPoisson1D(Poisson1DBase):
    def __init__(self, N, f, alpha, beta, gamma, delta, domain=(0, 1)):
        super().__init__(N, f, domain)
        self.alpha, self.beta = alpha, beta
        self.gamma, self.delta = gamma, delta
        self.create_matrix()

    def create_matrix(self):
        self.T = create_matrix(N=self.N, alpha=self.alpha, beta=self.beta, order=2)

    def solve(self):
        f_vec = self.f(self.x[1:-1]) * self.h**2
        f_vec[0] += self.gamma * self.h / (self.alpha * self.h + self.beta)
        f_vec[-1] += self.delta * self.h / (self.alpha * self.h + self.beta)
        
        u_int = la.solve(self.T, f_vec)
        
        self.u[1:-1] = u_int
        self.u[0] = (self.gamma * self.h + self.u[1] * self.beta) / (self.alpha * self.h + self.beta)
        self.u[-1] = (self.delta * self.h + self.u[-2] * self.beta) / (self.alpha * self.h + self.beta)
        
        return self.u
    

def main():

    def func(x):
        return -10 * np.ones_like(x)

    # solver = DirichletPoisson1D(N=10, f=func, u_a=0, u_b=0)
    solver = RobinPoisson1D(N=100, f=func, alpha=1, beta=0.5, gamma=0, delta=0)
    solver.solve()
    solver.plot_solution()
    

if __name__ == '__main__':
    main()