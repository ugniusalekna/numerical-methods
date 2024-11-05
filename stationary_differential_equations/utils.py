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

    T = np.zeros((N, N))
    
    for i in range(-half, half + 1):
        if i == 0:
            continue
        diag_values = np.full(N - abs(i), coeff[half + i])
        T += np.diag(diag_values, k=i)
    
    T += np.diag(np.full(N, coeff[half]))

    # correct only for order 2 and 4
    if alpha is not None and beta is not None:
        h = 1 / (N + 1)

        if order == 2:
            d = (2 * h * alpha + 3 * beta)

            if d != 0:
                T[0, 0] -= (4*beta / d)
                T[0, 1] += (beta / d)
                
                T[-1, -2] += (beta / d)
                T[-1, -1] -= (4*beta / d)
            else:
                print(f"Division by 0 in 'create_matrix'")

        elif order == 4:
            d = (12 * h * alpha + 25 * beta)
            
            if d != 0:
                T[0, 0] = 20 / 12 - 528 / 12 * beta / d
                T[0, 1] = -6 / 12 + 396 / 12 * beta / d
                T[0, 2] = -4 / 12 - 176 / 12 * beta / d
                T[0, 3] = 1 / 12 + 33 / 12 * beta / d
                
                T[1, 0] += 48 / 12 * beta / d
                T[1, 1] -= 36 / 12 * beta / d
                T[1, 2] += 16 / 12 * beta / d
                T[1, 3] -= 3 / 12 * beta / d                

                T[-1, -1] = 20 / 12 - 528 / 12 * beta / d
                T[-1, -2] = -6 / 12 + 396 / 12 * beta / d
                T[-1, -3] = -4 / 12 - 176 / 12 * beta / d
                T[-1, -4] = 1 / 12 + 33 / 12 * beta / d

                T[-2, -1] += 48 / 12 * beta / d
                T[-2, -2] -= 36 / 12 * beta / d
                T[-2, -3] += 16 / 12 * beta / d
                T[-2, -4] -= 3 / 12 * beta / d
            
            else:
                print(f"Division by 0 in 'create_matrix'")

        # T[0, 0] *= (alpha + beta / (2*h))
        # T[0, 1] *= (alpha + beta / h)
        
        # T[-1, -2] *= (alpha + beta / h)
        # T[-1, -1] *= (alpha + beta / (2*h))

    return T