import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')


def show_plot(x_data, y_data, 
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
                   figure_size=(10, 6),
                   font_size=12,
                   secondary_y_font_size=10,
                   secondary_y_color='tab:red'):
    
    fig, ax1 = plt.subplots(figsize=figure_size)
    
    if not isinstance(y_data, list):
        y_data = [y_data]
    if not isinstance(x_data, list):
        x_data = [x_data] * len(y_data)
    
    if labels is None:
        labels = [f"Series {i+1}" for i in range(len(y_data))]
    if line_styles is None:
        line_styles = ['-'] * len(y_data)
    if colors is None:
        colors = [None] * len(y_data)
    if markers is None:
        markers = [None] * len(y_data)
    if marker_sizes is None:
        marker_sizes = [6] * len(y_data)
    if line_widths is None:
        line_widths = [1.5] * len(y_data)
    if error_bars is None:
        error_bars = [None] * len(y_data)

    for i, (x, y) in enumerate(zip(x_data, y_data)):
        ax1.plot(x, y, 
                 label=labels[i],
                 linestyle=line_styles[i], 
                 color=colors[i], 
                 marker=markers[i], 
                 markersize=marker_sizes[i],
                 linewidth=line_widths[i])
        
        if error_bars[i] is not None:
            ax1.errorbar(x, y, yerr=error_bars[i], fmt='none', color=colors[i],
                         capsize=error_capsize, elinewidth=line_widths[i])

    if log_scale_x:
        ax1.set_xscale('log')
    if log_scale_y:
        ax1.set_yscale('log')
    
    if grid:
        ax1.grid(True, linestyle=grid_style, color=grid_color, alpha=grid_alpha)

    if x_label:
        ax1.set_xlabel(x_label, fontsize=font_size)
    if y_label:
        ax1.set_ylabel(y_label, fontsize=font_size)
    if title:
        ax1.set_title(title, fontsize=font_size + 2)

    if x_lim:
        ax1.set_xlim(x_lim)
    if y_lim:
        ax1.set_ylim(y_lim)
    
    if secondary_y_data is not None:
        ax2 = ax1.twinx()
        for y2 in secondary_y_data:
            ax2.plot(x_data[0], y2, color=secondary_y_color, linestyle='--')
        if y2_label:
            ax2.set_ylabel(y2_label, fontsize=secondary_y_font_size, color=secondary_y_color)
        if y2_lim:
            ax2.set_ylim(y2_lim)
        ax2.tick_params(axis='y', labelcolor=secondary_y_color)
    else:
        ax2 = None

    ax1.legend(loc=legend_loc, fontsize=font_size - 2)

    plt.tight_layout()
    plt.show()