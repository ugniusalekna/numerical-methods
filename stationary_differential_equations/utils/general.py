import numpy as np
import scipy.linalg as la


def create_matrix(N, delta=None, order=2):
    coeffs = {
        2: [-1.0, 2.0, -1.0],
        4: [1/12, -4/3, 5/2, -4/3, 1/12],
        6: [-1/90, 3/20, -3/2, 49/18, -3/2, 3/20, -1/90],
        8: [1/560, -8/315, 1/5, -8/5, 205/72, -8/5, 1/5, -8/315, 1/560],
        10: [-1/3150, 5/1008, -5/126, 5/21, -5/3, 5269/1800, -5/3, 5/21, -5/126, 5/1008, -1/3150]
    }

    if order not in coeffs:
        raise ValueError("No coeffs for this order of approximation.")

    coeff = coeffs[order]
    half = len(coeff) // 2

    h = 1 / (N + 1)
    T = np.zeros((N, N))
    
    for i in range(-half, half + 1):
        if i == 0:
            continue
        diag_values = np.full(N - abs(i), coeff[half + i])
        T += np.diag(diag_values, k=i)
    
    T += np.diag(np.full(N, coeff[half]))

    # only for order 2 and 4
    if order == 2:
        if delta is not None:
            p = (delta - 1) / (4 * h + 3 * (delta - 1))
            
        T[0, 0] -= 4 * p
        T[0, 1] += p
        
        T[-1, -2] += p
        T[-1, -1] -= 4 * p
        
    elif order == 4:
        if delta is not None:
            p = (delta - 1) / (24 * h + 25 * (delta - 1))
            
        T[0, 0] -= (5/6 + 44 * p)
        T[0, 1] += (5/6 + 33 * p)
        T[0, 2] -= (5/12 + 44/3 * p)
        T[0, 3] += (1/12 + 11/4 * p)
        
        T[1, 0] += 4 * p
        T[1, 1] -= 3 * p
        T[1, 2] += 4/3 * p
        T[1, 3] -= 1/4 * p
        
        T[-2, -4] -= 1/4 * p
        T[-2, -3] += 4/3 * p
        T[-2, -2] -= 3 * p
        T[-2, -1] += 4 * p
        
        T[-1, -4] += (1/12 + 11/4 * p)
        T[-1, -3] -= (5/12 + 44/3 * p)
        T[-1, -2] += (5/6 + 33 * p)
        T[-1, -1] -= (5/6 + 44 * p)
        
    return T


def solve_eigen(T):
    eigvals, eigvects = la.eig(T)
    eigvals = eigvals.real

    idxs = np.argsort(eigvals)

    eigvals = eigvals[idxs]
    eigvects = eigvects[:, idxs]

    for i in range(eigvects.shape[1]):
        if np.sum(eigvects[:, i]) < 0:
            eigvects[:, i] = -eigvects[:, i]

    return eigvals, eigvects
