import numpy as np


def create_matrix(N, alpha=None, beta=None, order=2):
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

    A = np.zeros((N, N))
    
    for i in range(-half, half + 1):
        if i == 0:
            continue
        diag_values = np.full(N - abs(i), coeff[half + i])
        A += np.diag(diag_values, k=i)
    
    A += np.diag(np.full(N, coeff[half]))

    if alpha is not None and beta is not None:
        h = 1 / (N + 1)
        d = (alpha * h + beta)
        if d != 0:
            A[0, 0] -= (beta / d)
            A[-1, -1] -= (beta / d)
        else:
            print("Division by zero in 'create_matrix'.")

    return A