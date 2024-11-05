import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import AxesWidget
from matplotlib import cbook
import scienceplots

plt.style.use('science')


def show_plot(ax, x_data, y_data, 
              labels=None, 
              title=None, 
              x_label=None, 
              y_label=None, 
              secondary_y_data=None, 
              y2_label=None,
              legend_loc="best", 
              line_styles=None, 
              colors=None, 
              markers=None, 
              marker_sizes=None,
              line_widths=None,
              log_scale_x=False, 
              log_scale_y=False,
              grid=True,
              grid_style='--',
              grid_color='gray',
              grid_alpha=0.5,
              x_lim=None,
              y_lim=None,
              y2_lim=None,
              error_bars=None, 
              error_capsize=3,
              font_size=12,
              secondary_y_font_size=10,
              secondary_y_color='tab:red',
              set_int_xticks=False):

    nested = isinstance(y_data[0], (list, np.ndarray))

    if not nested:
        x_data = [x_data]
        y_data = [y_data]

    n_series = len(y_data)

    def nest(attr, default_value):
        if attr is None:
            return [default_value] * n_series
        else:
            return [attr] * n_series

    labels = nest(labels, f"Series")
    line_styles = nest(line_styles, '-')
    colors = nest(colors, None)
    markers = nest(markers, None)
    marker_sizes = nest(marker_sizes, 6)
    line_widths = nest(line_widths, 1.5)
    error_bars = nest(error_bars, None)

    for i, (x, y) in enumerate(zip(x_data, y_data)):
 
        lnstl = line_styles[i]
        col = colors[i]
        mrk = markers[i]
        mrksz = marker_sizes[i]
        lnwd = line_widths[i]
        errb = error_bars[i]
        lbl = labels[i]

        use_scatter = isinstance(col, list) and len(col) == len(x)
        
        if use_scatter:
            used_labels = set()
            for j, (xi, yi) in enumerate(zip(x, y)):
                l = lbl[j] if lbl[j] not in used_labels else None
                if l is not None:
                    used_labels.add(l)
                ax.scatter(xi, yi, c=col[j], marker=mrk, s=mrksz, linewidth=lnwd, label=l)

        else:
            ax.plot(x, y, 
                    label=lbl,
                    linestyle=lnstl, 
                    color=col,
                    marker=mrk, 
                    markersize=mrksz,
                    linewidth=lnwd)

        if errb is not None:
            ax.errorbar(x, y, yerr=errb, fmt='none', color=col if not isinstance(col, list) else 'k',
                        capsize=error_capsize, elinewidth=lnwd)

    if log_scale_x:
        ax.set_xscale('log')
    if log_scale_y:
        ax.set_yscale('log')
    
    if grid:
        ax.grid(True, linestyle=grid_style, color=grid_color, alpha=grid_alpha)

    if x_label:
        ax.set_xlabel(x_label, fontsize=font_size)
    if y_label:
        ax.set_ylabel(y_label, fontsize=font_size)
    if title:
        ax.set_title(title, fontsize=font_size + 2)

    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)
    
    if secondary_y_data is not None:
        ax2 = ax.twinx()
        for y2 in secondary_y_data:
            ax2.plot(x_data[0], y2, color=secondary_y_color, linestyle='--')
        if y2_label:
            ax2.set_ylabel(y2_label, fontsize=secondary_y_font_size, color=secondary_y_color)
        if y2_lim:
            ax2.set_ylim(y2_lim)
        ax2.tick_params(axis='y', labelcolor=secondary_y_color)
    
    if labels:
        ax.legend(loc=legend_loc, fontsize=font_size - 2)
    
    if set_int_xticks:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))


class CustomRadioButtons(AxesWidget):
    def __init__(self, ax, labels, active=0, activecolor='blue', size=49,
                 orientation="vertical", label_spacing=0.1, **kwargs):
        """
        Add custom radio buttons to an `~.axes.Axes`.

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The axes to add the buttons to.
        labels : list of str
            The button labels.
        active : int
            The index of the initially selected button.
        activecolor : color
            The color of the selected button.
        size : float
            Size of the radio buttons.
        orientation : str
            The orientation of the buttons: 'vertical' (default), or 'horizontal'.
        label_spacing : float
            The spacing between the button markers and their labels.

        Further parameters are passed on to `Legend`.
        """
        AxesWidget.__init__(self, ax)
        self.activecolor = activecolor
        axcolor = ax.get_facecolor()
        self.value_selected = None

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_navigate(False)

        circles = []
        for i, label in enumerate(labels):
            if i == active:
                self.value_selected = label
                facecolor = activecolor
            else:
                facecolor = axcolor

            p = ax.scatter([], [], s=size, marker="o", edgecolor='black',
                           facecolor=facecolor)
            circles.append(p)

        if orientation == "horizontal":
            kwargs.update(ncol=len(labels), mode="expand")
        kwargs.setdefault("frameon", False)    

        self.box = ax.legend(circles, labels, loc="center", handletextpad=label_spacing, **kwargs)
        self.labels = self.box.texts
        self.circles = self.box.legend_handles

        for c in self.circles:
            c.set_picker(5)
        
        self._observers = cbook.CallbackRegistry()
        
        self.connect_event('pick_event', self._clicked)

    def _clicked(self, event):
        if (self.ignore(event) or event.mouseevent.button != 1 or
                event.mouseevent.inaxes != self.ax):
            return
        if event.artist in self.circles:
            self.set_active(self.circles.index(event.artist))

    def set_active(self, index):
        if 0 <= index < len(self.circles):
            for i, p in enumerate(self.circles):
                p.set_facecolor(self.activecolor if i == index else self.ax.get_facecolor())
            self.value_selected = self.labels[index].get_text()
            self._observers.process('clicked', self.value_selected)

    def on_clicked(self, func):
        """
        Connect a function to the button click event.

        Parameters
        ----------
        func : callable
            A function with a single parameter that will be called
            when a button is clicked. The parameter will be the text
            label of the clicked button.
        """
        self._observers.connect('clicked', func)