import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from matplotlib.widgets import Button, Slider

from plotting import show_plot, CustomRadioButtons


def create_matrix(N, alpha=None, beta=None, order=2):
    coeffs = {
        2: [-1.0, 2.0, -1.0],
        4: [1/12, -4/3, 5/2, -4/3, 1/12],
        6: [-1/90, 3/20, -3/2, 49/18, -3/2, 3/20, -1/90],
        8: [1/560, -8/315, 1/5, -8/5, 205/72, -8/5, 1/5, -8/315, 1/560]
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


def solve_eigen(A):
    eigvals, eigvects = la.eigh(A)

    for i in range(eigvects.shape[1]):
        if np.sum(eigvects[:, i]) < 0:
            eigvects[:, i] = -eigvects[:, i]

    return eigvals, eigvects


def plot_eigvals(eigvals, idx=None, ax=None):
    n = len(eigvals)
    colors = ['b'] * n
    if idx is not None:
        colors[idx] = 'r'

    ax.clear()
    show_plot(
        ax=ax,
        x_data=np.arange(1, n+1),
        y_data=[eigvals],
        labels=[r"Eigenvalues $\lambda_i$"],
        title=rf"Eigenvalues of matrix $T_{{{n}}}$",
        x_label=r"$i = 1,...,N$",
        y_label=r"$\lambda_i$",
        markers=['o'],
        line_styles=['none'],
        colors=[colors],
        legend_loc="upper left",
        set_int_xticks=True,
    )
    
    if ax is not None:
        plt.draw()


def plot_eigvects(eigvects, idx, ax=None):
    y_lim = (-1.0, 1.0)
    n = eigvects.shape[0]
    
    ax.clear()
    show_plot(
        ax=ax,
        x_data=np.arange(1, n+1),
        y_data=eigvects[:, idx],
        labels=[rf"Eigenvector $v_{{{idx+1}}}$ corresponding to $\lambda_{{{idx+1}}}$"],
        title=rf"Eigenvectors of matrix $T_{{{n}}}$",
        x_label=r"$j = 1,...,N$",
        y_label=rf"Coordinates $v^j_{{{idx+1}}}$",
        markers=['o'],
        line_styles=['-'],
        colors=[plt.get_cmap('tab10')((idx%10))],
        legend_loc="upper left",
        y_lim=y_lim,
        set_int_xticks=True,
    )
    
    if ax is not None:
        plt.draw()


def plot_with_slider(N):
    current_index = [0]
    current_order = [2]

    fig, (ax_vals, ax_vects) = plt.subplots(1, 2, figsize=(14, 7))
    fig.subplots_adjust(hspace=0.4, bottom=0.2)

    ax_slider_a = plt.axes([0.39, 0.08, 0.25, 0.02], facecolor='lightgray')
    slider_alpha = Slider(ax_slider_a, r'Parameter $\alpha$', -5.0, 5.0, valinit=1.0, valstep=0.1, color='blue')
    
    ax_slider_b = plt.axes([0.39, 0.05, 0.25, 0.02], facecolor='lightgray')
    slider_beta = Slider(ax_slider_b, r'Parameter $\beta$', -5.0, 5.0, valinit=0.0, valstep=0.1, color='blue')

    ax_radio_order = plt.axes([0.12, 0.05, 0.16, 0.05], facecolor='lightgray')
    radio_order =  CustomRadioButtons(ax_radio_order ,['2','4','6', '8'], active=0, orientation="horizontal", label_spacing=0.01)
    order_text = fig.text(0.12, 0.11, rf'Order of approximation $\mathcal{{O}}(h^{{{current_order[0]}}})$', fontsize=12, weight='bold')

    ax_next_button = plt.axes([0.81, 0.05, 0.1, 0.05])
    ax_prev_button = plt.axes([0.7, 0.05, 0.1, 0.05])

    next_button = Button(ax_next_button, 'Next')
    prev_button = Button(ax_prev_button, 'Prev')
    
    def update(val):
        alpha = slider_alpha.val
        beta = slider_beta.val
        order = current_order[0]
        
        A = create_matrix(N, alpha=alpha, beta=beta, order=order)
        
        global eigvals, eigvects
        eigvals, eigvects = solve_eigen(A)
        
        global num_eigvects
        num_eigvects = eigvects.shape[1]

        ax_vals.clear()
        ax_vects.clear()
        
        plot_eigvals(eigvals, idx=current_index[0], ax=ax_vals)
        plot_eigvects(eigvects, idx=current_index[0], ax=ax_vects)

    def update_order(label):
        current_order[0] = int(label)
        order_text.set_text(rf'Order of approximation $\mathcal{{O}}(h^{{{current_order[0]}}})$')
        update(None)

    def update_plot(forward=None):
        if forward is None:
            current_index[0] = 0
        else:
            step = 1 if forward else -1
            current_index[0] = (current_index[0] + step) % num_eigvects

        plot_eigvects(eigvects, idx=current_index[0], ax=ax_vects)
        plot_eigvals(eigvals, idx=current_index[0], ax=ax_vals)

    update(None) # init

    next_button.on_clicked(lambda event: update_plot(forward=True))
    prev_button.on_clicked(lambda event: update_plot(forward=False))

    slider_alpha.on_changed(update)
    slider_beta.on_changed(update)
    radio_order.on_clicked(update_order)

    plt.show()


def main():
    N = 101
    plot_with_slider(N)


if __name__ == '__main__':
    main()