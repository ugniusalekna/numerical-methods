import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from utils.plotting import show_plot, CustomRadioButtons
from utils.general import create_matrix, solve_eigen
np.set_printoptions(suppress=True, precision=3, floatmode='fixed', linewidth=150)


def plot_eigvals(eigvals, idx=None, ax=None, atol=1e-5, show_true=True):
    if ax is None:
        fig, ax = plt.subplots()

    n = len(eigvals)
    
    neg_eigvals = [val for val in eigvals if val < -atol]
    zero_eigvals = [val for val in eigvals if np.isclose(val, 0, atol=atol)]
    pos_eigvals = [val for val in eigvals if val > atol]
    
    x_vals_neg = np.arange(-len(neg_eigvals), 0)
    x_vals_zero = [0] if len(zero_eigvals) > 0 else []
    x_vals_pos = np.arange(1, len(pos_eigvals) + 1)

    x_data, y_data, labels, colors = [], [], [], []

    if len(neg_eigvals) > 0:
        x_data.extend(x_vals_neg)
        y_data.extend(neg_eigvals)
        labels.extend([r'$\lambda_i < 0$'] * len(neg_eigvals))
        colors.extend(['orange'] * len(neg_eigvals))

    if len(zero_eigvals) > 0:
        x_data.extend(x_vals_zero)
        y_data.extend(zero_eigvals)
        labels.extend([r'$\lambda_i \approx 0$'] * len(zero_eigvals))
        colors.extend(['g'] * len(zero_eigvals))
        
    if len(pos_eigvals) > 0:
        x_data.extend(x_vals_pos)
        y_data.extend(pos_eigvals)
        labels.extend([r'$\lambda_i > 0$'] * len(pos_eigvals))
        colors.extend(['b'] * len(pos_eigvals))

    if idx is not None:
        colors[idx] = 'r'
    
    neg_str = ", ".join([f"{i}" for i in x_vals_neg])
    rem_n = len(pos_eigvals)
    pos_str = f"1, \dots, {rem_n}" if rem_n > 1 else "1"

    if len(neg_eigvals) > 0 and len(zero_eigvals) > 0:
        x_label = rf"$i = {neg_str}, 0, {pos_str}$"
    elif len(neg_eigvals) > 0:
        x_label = rf"$i = {neg_str}, {pos_str}$"
    elif len(zero_eigvals) > 0:
        x_label = rf"$i = 0, {pos_str}$"
    else:
        x_label = rf"$i = {pos_str}$"

    ax.clear()
    show_plot(
        ax=ax,
        x_data=x_data,
        y_data=y_data,
        labels=labels,
        title=rf"Eigenvalues of matrix $T_{{{n}}}$",
        x_label=x_label,
        y_label=r"$\lambda_i$",
        colors=colors,
        markers='o',
        marker_sizes=6 - 5 * min(1, n / 200),
        legend_loc="upper left",
        set_int_xticks=True,
        x_lim=[-5, n+10],
        y_lim=[-2, 10],
    )

    if show_true:
        x_vals_theory = np.arange(1, len(pos_eigvals) + 1)
        y_vals_theory = (np.pi * x_vals_theory / (n + 1)) ** 2
        ax.plot(x_vals_theory, y_vals_theory, label=r'$\lambda = \left(\frac{\pi i}{N + 1}\right)^2$', linestyle='--', color='g')
    
    ax.legend(loc="upper left")

    if ax is not None:
        plt.draw()



def plot_eigvects(eigvects, eigvals, idx, ax=None, atol=1e-5):
    if ax is None:
        fig, ax = plt.subplots()
        
    y_lim = (-1.0, 1.0)
    n = eigvects.shape[0]

    neg_eigvals = [val for val in eigvals if val < -atol]
    zero_eigvals = [val for val in eigvals if np.isclose(val, 0, atol=atol)]

    neg_count = len(neg_eigvals)
    zero_count = len(zero_eigvals)
    
    if idx < neg_count:
        eig_idx = -neg_count + idx
    elif idx < neg_count + zero_count:
        eig_idx = 0
    else:
        eig_idx = idx - neg_count - zero_count + 1

    eig_label = rf"Eigenvector $v_{{{eig_idx}}}$ corresponding to $\lambda_{{{eig_idx}}}$"
    y_label = rf"Coordinates $v^j_{{{eig_idx}}}$"

    ax.clear()
    show_plot(
        ax=ax,
        x_data=np.arange(1, n + 1),
        y_data=eigvects[:, idx],
        labels=[eig_label],
        title=rf"Eigenvectors of matrix $T_{{{n}}}$",
        x_label=r"$j = 1,\dots,N$",
        y_label=y_label,
        markers='o',
        marker_sizes=6 - 5 * min(1, n / 200),
        line_styles='-',
        colors=plt.get_cmap('tab10')((idx % 10)),
        legend_loc="upper left",
        y_lim=y_lim,
        set_int_xticks=True,
    )
    
    if ax is not None:
        plt.draw()


class SpectralTool:
    def __init__(self, N):
        self.N = N
        self.curr_idx = 0
        self.curr_ord = 2

        self.eigvals = None
        self.eigvects = None
        self.num_eigvects = None

        self.init_fig()
        self.setup_ui()
        self.setup_buttons()
        
        self.update()  # init plot
        self.connect_callbacks()

    def init_fig(self):
        self.fig, (self.ax_vals, self.ax_vects) = plt.subplots(1, 2, figsize=(14, 7))
        self.fig.subplots_adjust(hspace=0.4, bottom=0.25)
    
    def clear_fig(self):
        self.ax_vals.clear()
        self.ax_vects.clear()
        
    def setup_ui(self):
        eq = r"$\begin{cases} L[u] = -u''(x) \\ B_1[u] = u(0) - \xi u'(0) \\ B_2[u] = u(1) + \xi u'(1) \end{cases}$"
        self.fig.text(0.39, 0.13, eq, fontsize=12, weight='bold')

        ax_slider_delta = plt.axes([0.39, 0.05, 0.25, 0.02], facecolor='lightgray')
        self.slider_delta = Slider(ax_slider_delta, r'$\delta = 2 \xi + 1$', -5.0, 5.0, valinit=1.0, valstep=0.1, color='blue')

        ax_radio_order = plt.axes([0.12, 0.05, 0.155, 0.05], facecolor='lightgray')
        self.radio_order = CustomRadioButtons(ax_radio_order, ['2', '4', '6', '8', '10'], active=0, orientation="horizontal", label_spacing=0.01)
        
        self.order_text = self.fig.text(0.12, 0.13, rf'Order of approximation $\mathcal{{O}}(h^{{{self.curr_ord}}})$', fontsize=12, weight='bold')

    def setup_buttons(self):
        ax_next_button = plt.axes([0.81, 0.05, 0.1, 0.05])
        ax_prev_button = plt.axes([0.7, 0.05, 0.1, 0.05])

        self.next_button = Button(ax_next_button, 'Next')
        self.prev_button = Button(ax_prev_button, 'Prev')

    def connect_callbacks(self):
        self.next_button.on_clicked(lambda e: self.update_eigvect_plot(forward=True))
        self.prev_button.on_clicked(lambda e: self.update_eigvect_plot(forward=False))
        self.slider_delta.on_changed(self.update)
        self.radio_order.on_clicked(self.update_order)

    def update(self, val=0.0):
        delta = self.slider_delta.val
        order = self.curr_ord
        
        T = create_matrix(self.N, delta=delta, order=order)
        self.eigvals, self.eigvects = solve_eigen(T)
        self.num_eigvects = self.eigvects.shape[1]
        
        self.clear_fig()
        
        plot_eigvals(self.eigvals, idx=self.curr_idx, ax=self.ax_vals)
        plot_eigvects(self.eigvects, self.eigvals, idx=self.curr_idx, ax=self.ax_vects)

    def update_order(self, label):
        self.curr_ord = int(label)
        self.order_text.set_text(rf'Order of approximation $\mathcal{{O}}(h^{{{self.curr_ord}}})$')
        self.update()

    def update_eigvect_plot(self, forward=True):
        step = 1 if forward else -1
        self.curr_idx = (self.curr_idx + step) % self.num_eigvects

        plot_eigvals(self.eigvals, idx=self.curr_idx, ax=self.ax_vals)
        plot_eigvects(self.eigvects, self.eigvals, idx=self.curr_idx, ax=self.ax_vects)

    def show(self):
        plt.show()
