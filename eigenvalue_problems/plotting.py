import numpy as np
import matplotlib.pyplot as plt


def plot_gershgorin_circles(A, eigvals=None):
    n = A.shape[0]
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for i in range(n):
        center = A[i, i]
        radius = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
        
        circle = plt.Circle((center.real, center.imag), radius, color='blue', fill=False, linestyle='--')
        ax.add_artist(circle)
        
        ax.plot(center.real, center.imag, 'bo', label="Center" if i == 0 else "")
    
    if eigvals is not None:
        ax.plot(eigvals.real, eigvals.imag, 'rx', label="Eigenvalues")

    max_radius = max(np.sum(np.abs(A[i, :])) - np.abs(A[i, i]) for i in range(n))
    xlims = (np.min(A.diagonal().real) - max_radius - 1, np.max(A.diagonal().real) + max_radius + 1)
    ylims = (np.min(A.diagonal().imag) - max_radius - 1, np.max(A.diagonal().imag) + max_radius + 1)
    
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_title("Gershgorin Circles")
    ax.axhline(0, color='black',linewidth=0.5)
    ax.axvline(0, color='black',linewidth=0.5)
    ax.legend()
    
    plt.show()