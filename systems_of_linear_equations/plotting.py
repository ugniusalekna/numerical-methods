import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')


def plot_omega_vs_iterations(solver, bounds, step, add_opt=False, plot_rho=False):    
    solver.collect = False
    omegas = np.arange(*bounds, step)
    num_iters = []
    rhos = []

    for omega in omegas:
        solver.reset()
        solver.omega = omega
        
        _ = solver.solve()
        
        if plot_rho:
            rhos.append(solver.get_spectral_radius(omega))
        
        num_iters.append(solver.iter+1)

    if add_opt:
        solver.reset()
        solver.omega = solver.get_omega_opt()
        _ = solver.solve()
        
        opt = [solver.omega, solver.iter+1]
        opt += [solver.get_spectral_radius(solver.omega)] if plot_rho else [] 

    fig, ax1 = plt.subplots(figsize=(8, 6))

    ax1.plot(omegas, num_iters, marker='o', linestyle='-', color='b', markersize=4)
    if add_opt:
        ax1.scatter(opt[0], opt[1], color='red', zorder=5, label=r"$\omega_{opt}$ = " + f"{opt[0]:.6f}")
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$\omega$', fontsize=10)
    ax1.tick_params(axis='y', colors='b')

    if plot_rho:
        ax2 = ax1.twinx()
        ax2.plot(omegas, rhos, marker='o', linestyle='-', color='g', markersize=4)
        if add_opt:
            ax2.scatter(opt[0], opt[2], color='orange', zorder=5, label=r"$\rho$ = " + f"{opt[2]:.6f}" + r" at $\omega_{opt}$")
        ax2.set_ylabel(r'$\rho$', fontsize=10, color='g')
        ax2.tick_params(axis='y', colors='g')

    fig.legend(loc="lower left", bbox_to_anchor=(0.075, 0.1))

    plt.title(r'$\omega$ vs. $n$' + (r' and $\rho$' if plot_rho else '') + ' (SOR Method)', fontsize=12)

    plt.tight_layout()
    plt.show()