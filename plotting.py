"""
This module contains all the functions needed to make the plots for the 
project 'DatadrivenLoadProfile'.

Description:
None

Notes:
The details of the project can be found at THIS-LINK.

Info:
- Author: G. Lorenti (gianmarco.lorenti@polito.it)
- Date: 21.05.2023
"""


# ----------------------------------------------------------------------------
# Import

# From Python libraries, packages, modules
import math
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import patheffects as pe
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# From self-created modules
from bill2watt.common.common import *


# ---------------------------------------------------------------------------
# Default settings
def_figsize = (3.5, 2)
def_fontsize = 10
color_map = {
    'tab:blue': 'Blues',
    'tab:red': 'Reds',
    'tab:green': 'Greens',
    'tab:orange': 'Oranges',
    'tab:purple': 'Purples'
}
def_colors = list(color_map.keys())
def_cmaps = list(color_map.values())


# ----------------------------------------------------------------------------
# Auxiliary functions

# Function to optimize the grid layout
def calculate_grid_layout(n, figsize):
    """Calculate the optimal number of rows and columns for a gridspec
    layout given the number of subplots and figure size."""

    # Calculate the optimal number of rows and columns
    nrows = math.isqrt(n)
    ncols = math.ceil(n / nrows)

    # Adjust the figure size to accommodate the subplots
    width, height = figsize
    width *= ncols
    height *= nrows

    # Return
    return nrows, ncols, (width, height)


# ----------------------------------------------------------------------------
# Function to create two concentric doughnut charts
# The two charts show the dataset distribution according to two different
# columns which are two-levels separation of the dataset.
def create_two_doughnuts(data: pd.DataFrame,
                         col_in: str,
                         col_out: str,
                         colors_in: list = None,
                         cmaps_out: list = None,
                         **kwargs: dict) -> plt.Figure:
    """
    Create two concentric doughnut charts.

    Description:
    The two charts show the dataset distribution according to two different
    columns which are two-levels separation of the dataset.

    Notes:
    None

    Parameters:
    - data (pandas DataFrame): Input data.
        The data shall contain the necessary columns.
    - col_in (str): Column name for the inner doughnut.
    - col_out (str): Column name for the outer doughnut.
    - colors_in (list): List of color names for the inner doughnut.
        If not provided, default colors are used. Optional, default is None.
    - cmaps_out (list): List of colormap names for the outer doughnut.
        If not provided, default colormaps are used. Optional, default is None.
    - **kwargs (dict): Additional keyword arguments for customization.

    Additional parameters:
    - size (float): Size of the inner doughnut.
        The size is relative to the outer doughnut. Default: 0.4.
    - alpha (float): Transparency of the doughnut charts.
        Default is 0.8.
    - radius (float): Radius of the doughnut charts.
        Default is 1.1.
    - figsize (tuple): Figure size.
        Default is (3.5, 2.5).
    - fontsize (int): Font size.
        Default is 10.
    - grid_spec (dict): GridSpec options for the figure.
        Default is dict(width_ratios=(1, 0.33)).
    - title (str): Title of the plot.
        If not provided, no title is generated. Optional, default is None.
    - title_pos (tuple): Position of the title.
        Value is relative to the figure. Default is (0.5, 0.95).

    Returns:
    - fig (matplotlib Figure): The generated figure.

    Info:
    - Author: G. Lorenti (gianmarco.lorenti@polito.it)
    - Date: 21.05.2023
    """

    # Extract additional parameters from kwargs or use default values
    figsize = kwargs.get('figsize', def_figsize)
    fontsize = kwargs.get('fontsize', def_fontsize)
    size = kwargs.get('size', 0.4)
    alpha = kwargs.get('alpha', 0.8)
    radius = kwargs.get('radius', 1.1)
    gridspec = kwargs.get('gridspec', dict(width_ratios=(1, 0.33)))
    title = kwargs.get('title', None)
    title_pos = kwargs.get('title_pos', (0.5, 0.95))
    # Check if the specified columns are in the data DataFrame
    assert col_in in data.columns,\
        f"Column '{col_in}' not found in the data DataFrame."
    assert col_out in data.columns,\
        f"Column '{col_out}' not found in the data DataFrame."

    # Extract values for inner and outer doughnuts
    data_in, data_out = \
        data.sort_values(by=[col_in, col_out])[[col_in, col_out]].values.T

    # Extract labels and counts for the inner doughnut
    labels_in = []
    counts_in = []
    for lab in data_in:
        if lab in labels_in:
            counts_in[labels_in.index(lab)] += 1
            continue
        labels_in.append(lab)
        counts_in.append(1)

    # Extract labels and counts for the outer doughnut
    labels_out = []
    counts_out = []
    for j, _ in enumerate(zip(labels_in, counts_in)):
        labels_out.append([])
        counts_out.append([])
        for lab in data_out[sum(counts_in[:j]):sum(counts_in[:j + 1])]:
            if lab in labels_out[j]:
                counts_out[j][labels_out[j].index(lab)] += 1
                continue
            labels_out[j].append(lab)
            counts_out[j].append(1)

    # Create figure
    fig, (ax, ax_leg) = plt.subplots(nrows=1, ncols=2, figsize=figsize,
                                     gridspec_kw=gridspec)

    # Extract colors for inner doughnut
    if colors_in is None:
        colors_in = [def_colors[i % len(def_colors)]
                     for i in range(len(labels_in))]
    else:
        assert len(colors_in) == len(labels_in), \
            f"The number of colors_in should match the number of labels_in."

    # Extract colors for outer doughnut
    if cmaps_out is None:
        cmaps_out = [color_map[c_in] for c_in in colors_in]
    else:
        assert len(cmaps_out) == len(labels_in), \
            f"The number of cmaps_out should match the number of labels_in."
    colors_out = [cm.get_cmap(cmap_out)(i)
                  for label_out, cmap_out in zip(labels_out, cmaps_out)
                  for i in np.linspace(0, 1, len(label_out))]

    # Adjust for plotting
    labels_out = [l for sub in labels_out for l in sub]
    counts_out = [v for sub in counts_out for v in sub]

    # Plot the doughnut charts
    ax.pie(counts_in, radius=radius - size, colors=colors_in,
           wedgeprops=dict(alpha=alpha, width=size, edgecolor='w'))
    ax.pie(counts_out, labels=labels_out, radius=radius, colors=colors_out,
           textprops=dict(fontsize=fontsize),
           wedgeprops=dict(alpha=alpha, width=size, edgecolor='w'))

    # Add legend in the dedicate axis
    ax_leg.axis('off')
    leg_elements = [Patch(label=l_in, color=c_in, alpha=alpha)
                    for l_in, c_in in zip(labels_in, colors_in)]
    ax_leg.legend(handles=leg_elements, fontsize=fontsize, loc='center')

    # Add a title
    fig.text(*title_pos, title, fontsize=fontsize, ha='center', va='center')

    # Return
    return fig


# ----------------------------------------------------------------------------
# Function to create aligned boxplot
# Show the distribution of the data according to a certain grouping
# and for different variables
def create_aligned_boxplots(data: pd.DataFrame,
                            groupby: str or list,
                            cols: str or list,
                            **kwargs: dict) -> plt.Figure:
    """
    Create aligned boxplots.

    Description:
    The aligned boxplots show the distribution of the data according to a
    certain grouping and for different variables.

    Notes:
    None

    Parameters:
    - data (pandas DataFrame): Input data.
        The data shall contain the necessary columns.
    - groupby (str or list): Column name(s) to group the data by.
    - cols (str or list): List of column names to create boxplots for.
    - title (str, optional): Title of the plot.
        Optional, default is None.
    - ylabel (str, optional): Label for the y-axis.
        Optional, default is None.
    - legend_labels (list, optional): List of labels for the legend.
        Optional, default is None.
    - **kwargs (dict): Additional keyword arguments for plot customization.

    Additional Parameters:
    - colors (list): List of colors for the boxplots.
        If not provided, default colors are used.
    - dx (float): Offset between boxplots along the x-axis. Default is 1.
    - whis (tuple): Percentiles to use for whiskers of the boxplots.
        Default is (5, 95).
    - figsize (tuple): Figure size. If not provided, default values are used.
    - fontsize (int): Font size. If not provided, default value is used.
    - horizontal (bool): Whether to create horizontal boxplots.
        Default is False.
    - gridspec (dict): Gridspec parameters for subplots.
        Default is dict(width_ratios=(1, 0.1)).
    - medianprops (dict): Properties for the medians.
        Default is dict(linewidth=3).
    - boxprops (dict): Properties for the boxes. Default is dict(alpha=0.5).
    - showfliers (bool): Whether to show outliers. Default is False.
    - title (str): Title of the figure. Default is None.
    - title_pos (tuple): Position of the title. Default is (0.5, 0.95).
    - xlabel (str): X-axis label.
        Only used if 'horizontal' is True. Default is None.
    - ylabel (str): Y-axis label.
        Only used if 'horizontal' is False. Default is None.
    - legend_labels (list): List of labels to use in the legend.
        They are related to the variables in the boxplot. If not provided,
        column names are used. Default is None.


    Returns:
    - fig (matplotlib Figure): The generated figure.

    Info:
    - Author: G. Lorenti (gianmarco.lorenti@polito.it)
    - Date: 21.05.2023
    """

    # Extract additional parameters from kwargs or use default values
    colors = kwargs.get('colors', def_colors)
    dx = kwargs.get('dx', 1)
    figsize = kwargs.get('figsize', def_figsize)
    fontsize = kwargs.get('fontsize', def_fontsize)
    horizontal = kwargs.get('horizontal', False)
    gridspec = kwargs.get('gridspec', dict(width_ratios=(1, 0.1)))
    whis = kwargs.get('whis', (5, 95))
    alpha = kwargs.get('alpha', 0.5)
    medianprops = kwargs.get('medianprops', dict(linewidth=3))
    boxprops = kwargs.get('boxprops', dict(alpha=0.5))
    showfliers = kwargs.get('showfliers', False)
    title = kwargs.get('title', None)
    title_pos = kwargs.get('title_pos', (0.5, 0.95))
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)
    legend_labels = kwargs.get('legend_labels', None)

    # Check if 'cols' and 'groupby' are in the data DataFrame
    cols = [cols] if isinstance(cols, str) else cols
    assert all(col in data.columns for col in cols), \
        "One or more columns in 'cols' are not found in the data DataFrame."
    groupby = [groupby] if isinstance(groupby, str) else groupby
    assert all(col in data.columns for col in groupby), \
        "One or more columns in 'groupby' are not found in the data DataFrame."

    # Group data and collect values
    values = [[] for _ in cols]
    names = []
    for i, (group, df) in enumerate(data.groupby(groupby)):
        names.append(group)
        for j, col in enumerate(cols):
            values[j].append(df[col].values.flatten())

    # Create positions of the boxplots
    positions = np.arange(1, len(names) + 1) * dx * (len(cols) + 1)

    # Create subplots
    fig, (ax, ax_leg) = plt.subplots(nrows=1, ncols=2, figsize=figsize,
                                     gridspec_kw=gridspec)
    # Create boxplots
    for idx, col_values in enumerate(values):
        medianprops = {**medianprops, **dict(color=colors[idx])}
        boxprops = {**boxprops, **dict(facecolor=colors[idx])}
        ax.boxplot(col_values, positions=positions + (idx - 1) * dx,
                   vert=not horizontal,
                   whis=whis, showfliers=showfliers, medianprops=medianprops,
                   patch_artist=True, boxprops=boxprops)

    # Add labels and ticks on the axes
    if horizontal:
        ax.set_yticks(positions, names, fontsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.grid(axis='x')
    else:
        ax.set_xticks(positions, names, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.grid(axis='y')
    ax.tick_params(labelsize=fontsize)

    # Add legend in the dedicated axis
    ax_leg.axis('off')
    leg_elements = \
        [Patch(label=label, alpha=alpha, color=color)
         for label, color
         in zip(cols if legend_labels is None else legend_labels, colors)]
    ax_leg.legend(handles=leg_elements, fontsize=fontsize, loc='center')

    # Add a title
    fig.text(*title_pos, title, fontsize=fontsize, ha='center', va='center')

    # Return
    return fig


# ----------------------------------------------------------------------------
def create_profile_percentiles(data: pd.DataFrame,
                               cols: list,
                               **kwargs):
    """
    Create profile of percentiles plot.

    Description:
    The 'profiles' are extracted from the data using the provided columns.
    The data provided are treated as samples, hence the percentiles are
    evaluated row-wise, for each column.

    Parameters:
    - data (pandas DataFrame): The input data.
    - cols (list): List of column names to plot.
    - **kwargs (dict): Additional keyword arguments for customization.

    Additional parameters:
    - q_min (float): Minimum quantile. Default is 0.05.
    - q_1 (float): First quartile. Default is 0.25.
    - q_med (float): Median quantile. Default is 0.5.
    - q_3 (float): Third quartile. Default is 0.75.
    - q_max (float): Maximum quantile. Default is 0.95.
    - figsize (tuple): Figure size. If not provided, default values are used.
    - fontsize (int): Font size. If not provided, default value is used.
    - gridspec (dict): Grid specifications.
        Default is dict(width_ratios=(1, 0.1)).
    - specs_min (dict): Specifications for minimum quantile line.
        Default is dict(color='tab:red', lw=1.5, ls='--').
    - specs_max (dict): Specifications for maximum quantile line.
        Default is dict(color='tab:red', lw=1.5, ls='-.').
    -specs_med (dict): Specifications for median quantile line.
        Default is dict(color='tab:blue', lw=1.5, patheffects=[...]).
    - specs_iqr (dict): Specifications for IQR (interquartile range) fill.
        Default is dict(color='gold', alpha=0.8).
    - title (str): Title of the figure. Default is None.
    - title_pos (tuple): Position of the title. Default is (0.5, 0.95).
    - xlabel (str): X-axis label. Default is None.
    - ylabel (str): Y-axis label. Default is None.
    - xticks (dict): X-axis tick parameters. Default is None.
    - yticks (dict): Y-axis tick parameters. Default is None.
    - grid (dict): Grid parameters. Default is None.

    Returns:
        fig (Figure): The generated matplotlib Figure object.
    """
    # Extract additional parameters from kwargs or use default values
    q_min = kwargs.get('q_min', 0.05)
    q_1 = kwargs.get('q_1', 0.25)
    q_med = kwargs.get('q_med', 0.5)
    q_3 = kwargs.get('q_3', 0.75)
    q_max = kwargs.get('q_max', 0.95)
    figsize = kwargs.get('figsize', (6, 4))
    fontsize = kwargs.get('fontsize', 10)
    gridspec = kwargs.get('gridspec', dict(width_ratios=(1, 0.1)))
    specs_min = kwargs.get('specs_min', dict(color='tab:red', lw=1.5, ls='--'))
    specs_max = kwargs.get('specs_max', dict(color='tab:red', lw=1.5, ls='--'))
    specs_med = \
        kwargs.get('specs_med',
                   dict(color='tab:blue', lw=1.5,
                        path_effects=[pe.Stroke(foreground='w', linewidth=3),
                                      pe.Normal()]))
    specs_iqr = kwargs.get('specs_iqr', dict(color='gold', alpha=0.8))
    title = kwargs.get('title', None)
    title_pos = kwargs.get('title_pos', (0.5, 0.95))
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)
    xticks= kwargs.get('xticks', None)
    yticks = kwargs.get('yticks', None)
    grid = kwargs.get('grid', None)

    # Values of the power and of the different quantiles
    powers = data[cols].values
    q_mins, q_1s, q_meds, q_3s, q_maxs = \
        np.quantile(powers, [q_min, q_1, q_med, q_3, q_max], axis=0)

    # Create subplots
    fig, (ax, ax_leg) = plt.subplots(1, 2, figsize=figsize,
                                     gridspec_kw=gridspec)

    # Plot quantiles
    t = np.arange(len(cols))
    ax.plot(t, q_mins, **specs_min)
    ax.fill_between(t, q_1s, q_3s, **specs_iqr)
    ax.plot(t, q_maxs, **specs_max)
    ax.plot(t, q_meds, **specs_med)

    # Add labels
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if xticks is not None:
        ax.set_xticks(**xticks)
    if yticks is not None:
        ax.set_yticks(**yticks)
    if grid is not None:
        ax.grid(**grid)
    ax.tick_params(labelsize=fontsize)

    # Add legend in dedicated axis
    ax_leg.axis('off')
    leg_elements = \
        [Line2D([0], [0], label=label, **specs) for label, specs
         in zip(('Min', 'Med', 'Max'),
                (specs_min, specs_med, specs_max))]
    leg_elements.append(Patch(label='IQR', **specs_iqr))
    ax_leg.legend(handles=leg_elements, fontsize=fontsize, loc='center')

    # Add title
    fig.text(*title_pos, title, fontsize=fontsize, ha='center', va='center')

    # Return
    return fig

