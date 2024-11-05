import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from matplotlib.widgets import Button, Slider

from plotting import show_plot, CustomRadioButtons
from utils import create_matrix


def solve_eigen(T):
    eigvals, eigvects = la.eigh(T)

    for i in range(eigvects.shape[1]):
        if np.sum(eigvects[:, i]) < 0:
            eigvects[:, i] = -eigvects[:, i]

    return eigvals, eigvects


def plot_eigvals(eigvals, idx=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    n = len(eigvals)

    negative_eigvals = [val for val in eigvals if val < 0]
    zero_eigvals = [val for val in eigvals if np.isclose(val, 0, atol=1e-5)]
    positive_eigvals = [val for val in eigvals if val > 0]

    x_vals_neg = np.arange(1, len(negative_eigvals) + 1)
    x_vals_zero = np.arange(len(negative_eigvals) + 1, len(negative_eigvals) + len(zero_eigvals) + 1)
    x_vals_pos = np.arange(len(negative_eigvals) + len(zero_eigvals) + 1, n + 1)

    x_data, y_data, labels, colors = [], [], [], []

    if len(negative_eigvals) > 0:
        x_data.extend(x_vals_neg)
        y_data.extend(negative_eigvals)
        labels.extend([r'$\lambda_i < 0$'] * len(negative_eigvals))
        colors.extend(['orange'] * len(negative_eigvals))

    if len(zero_eigvals) > 0:
        x_data.extend(x_vals_zero)
        y_data.extend(zero_eigvals)
        labels.extend([r'$\lambda_i = 0$'] * len(zero_eigvals))
        colors.extend(['g'] * len(zero_eigvals))
        
    if len(positive_eigvals) > 0:
        x_data.extend(x_vals_pos)
        y_data.extend(positive_eigvals)
        labels.extend([r'$\lambda_i > 0$'] * len(positive_eigvals))
        colors.extend(['b'] * len(positive_eigvals))

    if idx is not None:
        colors[idx] = 'r'
    
    ax.clear()
    show_plot(
        ax=ax,
        x_data=x_data,
        y_data=y_data,
        labels=labels,
        title=rf"Eigenvalues of matrix $T_{{{n}}}$",
        x_label=r"$i = 1,\dots,N$",
        y_label=r"$\lambda_i$",
        colors=colors,
        markers='o',
        marker_sizes=6 - 5 * min(1, n / 200),
        legend_loc="upper left",
        set_int_xticks=True,
        x_lim=[-10, n+10],
        y_lim=[-10, 10],
    )

    x_vals_theory = np.arange(1, n + 1)
    y_vals_theory = (np.pi * x_vals_theory / (n + 1)) ** 2
    ax.plot(x_vals_theory, y_vals_theory, label=r'$\lambda = \left(\frac{\pi i}{N+1}\right)^2$', linestyle='--', color='g')
    ax.legend(loc="upper left")

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
        x_label=r"$j = 1,\dots,N$",
        y_label=rf"Coordinates $v^j_{{{idx+1}}}$",
        markers='o',
        marker_sizes=6 - 5 * min(1, n / 200),
        line_styles='-',
        colors=plt.get_cmap('tab10')((idx%10)),
        legend_loc="upper left",
        y_lim=y_lim,
        set_int_xticks=True,
    )
    
    if ax is not None:
        plt.draw()


def visualize_spectrum(N):
    current_index = [0]
    current_order = [2]

    fig, (ax_vals, ax_vects) = plt.subplots(1, 2, figsize=(14, 7))
    fig.subplots_adjust(hspace=0.4, bottom=0.2)

    ax_slider_a = plt.axes([0.39, 0.08, 0.25, 0.02], facecolor='lightgray')
    slider_alpha = Slider(ax_slider_a, r'Parameter $\alpha$', -5.0, 5.0, valinit=1.0, valstep=0.1, color='blue')
    
    ax_slider_b = plt.axes([0.39, 0.05, 0.25, 0.02], facecolor='lightgray')
    slider_beta = Slider(ax_slider_b, r'Parameter $\beta$', -5.0, 5.0, valinit=0.0, valstep=0.1, color='blue')

    ax_radio_order = plt.axes([0.12, 0.05, 0.155, 0.05], facecolor='lightgray')
    radio_order =  CustomRadioButtons(ax_radio_order, ['2','4','6','8', '10'], active=0, orientation="horizontal", label_spacing=0.01)
    order_text = fig.text(0.12, 0.11, rf'Order of approximation $\mathcal{{O}}(h^{{{current_order[0]}}})$', fontsize=12, weight='bold')

    ax_next_button = plt.axes([0.81, 0.05, 0.1, 0.05])
    ax_prev_button = plt.axes([0.7, 0.05, 0.1, 0.05])

    next_button = Button(ax_next_button, 'Next')
    prev_button = Button(ax_prev_button, 'Prev')
    
    def update(val=0.0):
        alpha = slider_alpha.val
        beta = slider_beta.val
        order = current_order[0]
        
        T = create_matrix(N, alpha=alpha, beta=beta, order=order)
        
        global eigvals, eigvects
        eigvals, eigvects = solve_eigen(T)
        
        global num_eigvects
        num_eigvects = eigvects.shape[1]

        ax_vals.clear()
        ax_vects.clear()
        
        plot_eigvals(eigvals, idx=current_index[0], ax=ax_vals)
        plot_eigvects(eigvects, idx=current_index[0], ax=ax_vects)

    def update_order(label):
        current_order[0] = int(label)
        order_text.set_text(rf'Order of approximation $\mathcal{{O}}(h^{{{current_order[0]}}})$')
        update()

    def update_plot(forward):
        step = 1 if forward else -1
        current_index[0] = (current_index[0] + step) % num_eigvects

        plot_eigvects(eigvects, idx=current_index[0], ax=ax_vects)
        plot_eigvals(eigvals, idx=current_index[0], ax=ax_vals)

    update() # init

    next_button.on_clicked(lambda e: update_plot(forward=True))
    prev_button.on_clicked(lambda e: update_plot(forward=False))

    slider_alpha.on_changed(update)
    slider_beta.on_changed(update)
    radio_order.on_clicked(update_order)

    plt.show()


def main():
    N = 5
    visualize_spectrum(N)
    

if __name__ == '__main__':
    main()