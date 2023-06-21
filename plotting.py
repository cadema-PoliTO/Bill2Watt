"""


Notes
-----

Info
----
Author: G. Lorenti
Email: gianmarco.lorenti@polito.it
"""

import math

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import patheffects as pe
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from bill2watt.common.common import *


# Default colors and colormaps
color_map = {
    'tab:blue': 'Blues',
    'tab:red': 'Reds',
    'tab:green': 'Greens',
    'tab:orange': 'Oranges',
    'tab:purple': 'Purples'
}
def_colors = list(color_map.keys())
def_cmaps = list(color_map.values())


# Auxiliary functions


def calculate_grid_layout(n, figsize):
    """
    Calculate the optimal number of rows and columns for a gridspec layout
    given the number of subplots and figure size.

    Parameters
    ----------
    n : int
        Number of subplots.
    figsize : tuple
        Figure size in inches (width, height).

    Returns
    -------
    int
        Number of rows in the layout.
    int
        Number of columns in the layout.
    tuple
        Adjusted figure size to accommodate the subplots.
    """

    # Calculate the optimal number of rows and columns
    nrows = math.isqrt(n)
    ncols = math.ceil(n / nrows)

    # Adjust the figure size to accommodate the subplots
    width, height = figsize
    width *= ncols
    height *= nrows

    return nrows, ncols, (width, height)


def get_rc_params(**kwargs):
    """
    Get the Matplotlib RC parameters from the provided keyword arguments.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments that can be used to update matplotlib.rc_params.

    Returns
    -------
    dict
        Matplotlib RC parameters.
    """
    rc_params = {key: param for key, param in kwargs.items() if
                 key in matplotlib.rcParams}
    return rc_params


# Plotting facilities


def create_two_doughnuts(data, col_in, col_out, colors_in=None, cmaps_out=None,
                         **kwargs):
    """
    Create two concentric doughnut charts, showing a dataset distribution
    according to two different columns which are two-levels separation of the
    dataset.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data.
    col_in : str
        Column name for the inner doughnut.
    col_out : str
        Column name for the outer doughnut.
    colors_in : list, optional
        List of color names for the inner doughnut, by default None.
    cmaps_out : list, optional
        List of colormap names for the outer doughnut, by default None.
    **kwargs : dict
        Additional keyword arguments for customization.

    Additional parameters
    ---------------------
    size : float, optional
        Size of the inner doughnut, relative to the outer doughnut.
    alpha : float, optional
        Transparency of the doughnut charts.
    radius : float, optional
        Radius of the doughnut charts.
    grid_spec : dict, optional
        GridSpec options for the figure.
    title : str, optional
        Title of the plot.
    title_pos : tuple, optional
        Position of the title.

    Returns
    -------
    plt.Figure
        The generated figure.
    """

    # Get keyword arguments or use default values
    size = kwargs.get('size', 0.4)
    alpha = kwargs.get('alpha', 0.8)
    radius = kwargs.get('radius', 1.1)
    gridspec = kwargs.get('gridspec', dict(width_ratios=(1, 0.33)))
    title = kwargs.get('title', None)
    title_pos = kwargs.get('title_pos', (0.5, 0.95))

    # Eventually update rc_params from 'kwargs'
    plt.rcParams.update(get_rc_params(**kwargs))

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
    fig, (ax, ax_leg) = plt.subplots(nrows=1, ncols=2, gridspec_kw=gridspec)

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
           wedgeprops=dict(alpha=alpha, width=size, edgecolor='w'))

    # Add legend in the dedicate axis
    ax_leg.axis('off')
    leg_elements = [Patch(label=l_in, color=c_in, alpha=alpha)
                    for l_in, c_in in zip(labels_in, colors_in)]
    ax_leg.legend(handles=leg_elements, loc='center')

    # Add a title
    fig.text(*title_pos, title, ha='center', va='center')

    return fig


def create_aligned_boxplots(data, groupby, cols, **kwargs):
    """
    Create aligned boxplots showing the distribution of the data according to
    a certain grouping and for different variables.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    groupby : str or list
        Column name(s) for grouping.
    cols : str or list
        Column name(s) for variables.

    Additional parameters
    ---------------------
    colors : list, optional
        List of colors for the boxplots.
    dx : int, optional
        Spacing between boxplots.
    horizontal : bool, optional
        Orientation of the boxplots.
    gridspec : dict, optional
        GridSpec options for the figure.
    whis : tuple, optional
        Percentiles for whiskers calculation.
    alpha : float, optional
        Transparency of the boxes.
    medianprops : dict, optional
        Properties for the median line.
    boxprops : dict, optional
        Properties for the box.
    showfliers : bool, optional
        Whether to show the outliers.
    title : str, optional
        Title of the plot.
    title_pos : tuple, optional
        Position of the title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    legend_labels : list, optional
        Custom labels for the legend.

    Returns
    -------
    plt.Figure
        The generated figure.
    """

    # Get keyword arguments or use default values
    colors = kwargs.get('colors', def_colors)
    dx = kwargs.get('dx', 1)
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

    # Eventually update rc_params from 'kwargs'
    plt.rcParams.update(get_rc_params(**kwargs))

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
    fig, (ax, ax_leg) = plt.subplots(nrows=1, ncols=2, gridspec_kw=gridspec)
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
        ax.set_yticks(positions, names)
        ax.set_xlabel(xlabel)
        ax.grid(axis='x')
    else:
        ax.set_xticks(positions, names)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y')

    # Add legend in the dedicated axis
    ax_leg.axis('off')
    leg_elements = \
        [Patch(label=label, alpha=alpha, color=color)
         for label, color
         in zip(cols if legend_labels is None else legend_labels, colors)]
    ax_leg.legend(handles=leg_elements, loc='center')

    # Add a title
    fig.text(*title_pos, title, ha='center', va='center')

    return fig


def create_profile_percentiles(data, cols, **kwargs):
    """
    Create profile of percentiles plot, which are extracted from the data
    using the provided columns. The data provided are treated as samples,
    hence the percentiles are evaluated row-wise, for each column.

    Parameters
    ----------
    data : DataFrame
        Input data containing the profile samples.
    cols : list
        List of column names in the data to use for percentile evaluation.

    Additional parameters
    ---------------------
    q_min : float, optional
        The minimum quantile value to plot.
    q_1 : float, optional
        The 25th percentile value to plot.
    q_med : float, optional
        The median (50th percentile) value to plot.
    q_3 : float, optional
        The 75th percentile value to plot.
    q_max : float, optional
        The maximum quantile value to plot.
    specs_min : dict, optional
        Specifications for the line plot of the minimum quantile.
    specs_max : dict, optional
        Specifications for the line plot of the maximum quantile
        (default: {'color': 'tab:red', 'lw': 1.5, 'ls': '--'}).
    specs_med : dict, optional
        Specifications for the line plot of the median.
    specs_iqr : dict, optional
        Specifications for the fill between the 25th and 75th percentiles.
    title : str, optional
        Title of the plot.
    title_pos : tuple, optional
        Position of the title in the figure.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    xticks : dict, optional
        Specifications for customizing x-axis ticks.
    yticks : dict, optional
        Specifications for customizing y-axis ticks.
    grid : dict, optional
        Specifications for grid display.
    gridspec : dict, optional
        Specifications for the gridspec layout.

    Returns
    -------
    plt.Figure
        The generated figure.
    """

    # Get keyword arguments or use default values
    q_min = kwargs.get('q_min', 0.05)
    q_1 = kwargs.get('q_1', 0.25)
    q_med = kwargs.get('q_med', 0.5)
    q_3 = kwargs.get('q_3', 0.75)
    q_max = kwargs.get('q_max', 0.95)
    specs_min = kwargs.get('specs_min', dict(color='tab:red', lw=1.5, ls='--'))
    specs_max = kwargs.get('specs_max', dict(color='tab:red', lw=1.5, ls='--'))
    specs_med = kwargs.get('specs_med', dict(
        color='tab:blue', lw=1.5, path_effects=[
            pe.Stroke(foreground='w', linewidth=3), pe.Normal()]))
    specs_iqr = kwargs.get('specs_iqr', dict(color='gold', alpha=0.8))
    title = kwargs.get('title', None)
    title_pos = kwargs.get('title_pos', (0.5, 0.95))
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)
    xticks= kwargs.get('xticks', None)
    yticks = kwargs.get('yticks', None)
    grid = kwargs.get('grid', None)
    gridspec = kwargs.get('gridspec', dict(width_ratios=(1, 0.1)))

    # Eventually update rc_params from 'kwargs'
    plt.rcParams.update(get_rc_params(**kwargs))

    # Values of the power and of the different quantiles
    powers = data[cols].values
    q_mins, q_1s, q_meds, q_3s, q_maxs = \
        np.quantile(powers, [q_min, q_1, q_med, q_3, q_max], axis=0)

    # Create subplots
    fig, (ax, ax_leg) = plt.subplots(1, 2, gridspec_kw=gridspec)

    # Plot quantiles
    t = np.arange(len(cols))
    ax.plot(t, q_mins, **specs_min)
    ax.fill_between(t, q_1s, q_3s, **specs_iqr)
    ax.plot(t, q_maxs, **specs_max)
    ax.plot(t, q_meds, **specs_med)

    # Add labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xticks is not None:
        ax.set_xticks(**xticks)
    if yticks is not None:
        ax.set_yticks(**yticks)
    if grid is not None:
        ax.grid(**grid)

    # Add legend in dedicated axis
    ax_leg.axis('off')
    leg_elements = \
        [Line2D([0], [0], label=label, **specs) for label, specs
         in zip(('Min', 'Med', 'Max'),
                (specs_min, specs_med, specs_max))]
    leg_elements.append(Patch(label='IQR', **specs_iqr))
    ax_leg.legend(handles=leg_elements, loc='center')

    # Add title
    fig.text(*title_pos, title, ha='center', va='center')

    return fig


def create_radar_plot(data, labels=None, xticklabels=None, **kwargs):
    """
     Create a radar plot based on the provided data.

     Parameters
     ----------
     data : pd.DataFrame
         DataFrame containing the radar plot values.
     labels : list, optional
         List of labels for each row in the data.
     xticklabels : list, optional
         List of labels for the radar plot x-axis.
     **kwargs : dict
         Additional keyword arguments for customizing the plot.

     Additional parameters
     ---------------------
     linewidth : float, optional
         The linewidth of the radar plot lines.
     xlim : list, optional
         The limits for the x-axis.
     gridspec_kw : dict, optional
         Specifications for the gridspec layout.
     title : str, optional
         Title of the radar plot.

     Returns
     -------
     matplotlib.figure.Figure
         The generated radar plot figure.
     """

    # Get keyword arguments or use default values
    linewidth = kwargs.get('linewidth', 1)
    xlim = kwargs.get('xlim', [None, None])
    gridspec_kw = kwargs.get('gridspec_kw', dict(width_ratios=[0.5, 0.5]))
    title = kwargs.get('title', '')

    # Eventually update rc_params from 'kwargs'
    plt.rcParams.update(get_rc_params(**kwargs))

    # Get the columns of the dataframe (radars)
    columns = data.columns.tolist()

    # Get labels (of legend) from data if not provided
    if labels is None:
        labels = list(data.index)

    # Get xticklabels from data if not provided
    if xticklabels is None:
        xticklabels = columns

    # Generate evenly spaced angles for the radar plot
    angles = np.linspace(0, 2 * np.pi, len(columns), endpoint=False).tolist()
    angles += angles[:1]  # Close the plot

    # Create subplots
    fig, (ax, ax_leg) = plt.subplots(nrows=1, ncols=2,
                                     subplot_kw={'polar': True},
                                     gridspec_kw=gridspec_kw)

    # Plot the radar lines for each test
    for i, (_, row) in enumerate(data.iterrows()):
        values = row.values
        values = np.append(values, values[0])  # Close the plot

        # Plot the radar lines
        ax.plot(angles, values, linewidth=linewidth, label=labels[i])

    # Set the X axis labels, ticks and lims
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(xticklabels)
    ax.set_xlim(xlim)

    # Set the radar plot title
    ax.set_title(title)

    # Add a legend plot
    legend_elements = ax.get_legend_handles_labels()
    ax_leg.axis('off')
    ax_leg.legend(*legend_elements)

    # Adjust the figure layout
    fig.tight_layout()

    return fig


def create_row_boxplots(data, xticklabels=None, **kwargs):
    """
    Create row-based boxplots using the provided data.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the boxplot values.

    xticklabels : list, optional
        List of labels for the x-axis tick marks.

    **kwargs : dict
        Additional keyword arguments for customizing the plot.

    Additional parameters
    ---------------------
    whis : float or sequence, optional
        The whisker positions in units of data width.
    showfliers : bool, optional
        Whether to show the outliers.
    showmeans : bool, optional
        Whether to show the means.
    boxprops : dict, optional
        Properties for customizing the box appearance.
    medianprops : dict, optional
        Properties for customizing the median line appearance.
    meanprops : dict, optional
        Properties for customizing the mean line appearance.
    title : str, optional
        Title of the plot.
    ylabel : str, optional
        Label for the y-axis.

    Returns
    -------
    matplotlib.figure.Figure
        The generated boxplot figure.
    """

    # Get keyword arguments or use default values
    whis = kwargs.get('whis', (0.05, 0.95))
    showfliers = kwargs.get('showfliers', False)
    showmeans = kwargs.get('showmeans', False)
    boxprops = kwargs.get('boxprops', dict(facecolor='tab:blue'))
    whiskerprops= kwargs.get('whiskerprops', dict())
    medianprops = kwargs.get('medianprops', dict(color='tab:orange'))
    meanprops = kwargs.get('meanprops', dict(color='tab:green'))
    title = kwargs.get('title', '')
    ylabel = kwargs.get('ylabel', '')

    # Eventually update rc_params from 'kwargs'
    plt.rcParams.update(get_rc_params(**kwargs))

    # Get xticklabels from data if not provided
    if xticklabels is None:
        xticklabels = list(data.index)

    # Set the positions for the boxes
    positions = range(len(data))

    # Get the values for each position
    values = list(data.values)

    # Create subplots
    fig, ax = plt.subplots()

    # Plot the boxplots
    ax.boxplot(values, positions=positions, whis=whis,
               showfliers=showfliers, showmeans=showmeans, patch_artist=True,
               boxprops=boxprops,whiskerprops=whiskerprops,
               medianprops=medianprops, meanprops=meanprops)

    # Set the x-axis ticks and xticklabels
    ax.set_xticks(positions)
    ax.set_xticklabels(xticklabels)

    # Set the y label
    ax.set_ylabel(ylabel)

    # Set the plot title
    ax.set_title(title)

    # Adjust the figure layout
    fig.tight_layout()

    return fig
