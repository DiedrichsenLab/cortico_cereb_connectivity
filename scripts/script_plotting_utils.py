import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from surfplot import Plot
from neuromaps.datasets import fetch_fslr
import nitools as nt
import numpy as np


def get_color_palette():
    palette = sns.diverging_palette(240, 10, n=2)
    color_palette = dict()
    color_palette['L2reg'] = palette[1]
    color_palette['NNLS'] = palette[0]
    return color_palette
# purple: #805173
# pink: #dcc2d7
# green: #327078
# olive: #95a389
# orange: #bc9553


def plot_heatmap_annotate(df, x_order, y_order, fig=None,
                          column=['train_dataset'], row=['eval_dataset'], value=['R_eval_adj'],
                          cmap='rocket', vmin=0, vmax=1, cbar=True,
                          ax=None, linewidths=0.5, annot_kws={"fontsize": 7}):
    
    # create heatmap from the dataframe
    V = pd.pivot_table(df, columns=column, index=row, values=value)
    V = V.reindex(y_order, axis=0)
    V = V.reindex(x_order, level=1, axis=1)

    if fig is None:
        fig = plt.figure(figsize=(9, 7.2))

    # make two separate axes for dataset vs global
    if len(x_order) != len(y_order):
        n = len(y_order)
        extra = len(x_order) - n

        gs = GridSpec(1, 2, width_ratios=[n, extra], wspace=0.05)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax2.sharey(ax1)

        sns.heatmap(V.values[:, :n], annot=True, fmt=".2f", cmap=cmap, vmin=vmin, vmax=vmax, 
                    xticklabels=V.columns.get_level_values(1).values[:n], 
                    yticklabels=V.index.values, square=True, ax=ax1, cbar=False, linewidths=linewidths, annot_kws=annot_kws)
        
        sns.heatmap(V.values[:, n:], annot=True, fmt=".2f", cmap=cmap, vmin=vmin, vmax=vmax, 
                xticklabels=V.columns.get_level_values(1).values[n:], 
                yticklabels=V.index.values, square=True, ax=ax2, cbar=False, linewidths=linewidths, annot_kws=annot_kws)

        ax1.tick_params(axis="x", labelrotation=90)
        ax2.tick_params(axis="x", labelrotation=90)
        ax2.tick_params(left=False, labelleft=False)

        # add cbar
        if cbar:
            cbar_ax = fig.colorbar(ax2.collections[0], ax=[ax1, ax2], orientation="vertical", fraction=0.035, pad=0.04)
            cbar_ax.outline.set_visible(False)
        else:
            cbar_ax = None

    else:
        sns.heatmap(V.values, annot=True, fmt=".2f", cmap=cmap, vmin=vmin, vmax=vmax, 
                    xticklabels=V.columns.get_level_values(1).values, 
                    yticklabels=V.index.values, square=True, ax=ax, cbar=cbar, linewidths=linewidths)

    return fig, ax1, ax2, cbar_ax

def plot_barplot_with_error(df_all, dots='strip', palette=get_color_palette(), alpha1=0.2, alpha2=1):
    ax = sns.barplot(data=df_all, x='train_dataset', y='R_eval_adj', hue='method',
                errorbar='se', alpha=alpha2, palette=palette, saturation=1.0, gap=0.05)
    
    if dots == 'strip':
        sns.stripplot(data=df_all, x='train_dataset', y='R_eval_adj', hue='method',
                      dodge=True, alpha=alpha1, marker='o', size=1.5, legend=False, palette='dark:black')
    elif dots == 'swarm':
        sns.swarmplot(data=df_all, x='train_dataset', y='R_eval_adj', hue='method',
                      dodge=True, alpha=alpha1, marker='o', size=1.5, legend=False, palette='dark:black')
        
    return ax


def plot_boxplot_strip(df, width=0.8, gap=0.4, linewidth=1, alpha=0.3, palette=get_color_palette()):
    ax = sns.boxplot(df, x='train_dataset', y='R_eval_adj', hue='method',
                width=width, gap=gap, linewidth=linewidth, showfliers=False, palette=palette)

    for patch in ax.patches:
        patch.set_alpha(alpha-0.1)
        patch.set_edgecolor(patch.get_facecolor())
        patch.set_linewidth(linewidth)

    for line in ax.lines:
        line.set_alpha(1.0)
        line.set_linewidth(linewidth)

    sns.stripplot(df, x='train_dataset', y='R_eval_adj', hue='method',
                  dodge=True, alpha=alpha, marker='o', size=2, legend=False, palette=palette)
    
    return ax
    

def plot_emp_CDF(x_r, y_r, x_n, y_n, palette=get_color_palette()):
    plt.plot(x_r, y_r, color=palette['L2reg'])
    plt.plot(x_n, y_n, color=palette['NNLS'])
    ax = plt.gca()

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=0.8)

    return ax


def plot_cortex_map(cifti_img,
                    layout='grid', figsize=None,
                    threshold=None,
                    cbar=True, cmap='seismic', cscale=None, alpha=0.7, zero_transparent=False):

    surfaces = fetch_fslr()

    lh, rh = surfaces['inflated']
    sulc_lh, sulc_rh = surfaces['sulc']

    if layout == 'grid':
        size = (500, 400)
    elif layout == 'row':
        size = (1100, 200)
    elif layout == 'column':
        size = (200, 1100)

    p = Plot(lh, rh, layout=layout, size=size)

    # cortical shading
    p.add_layer(
        {'left': sulc_lh, 'right': sulc_rh},
        cmap='binary_r',
        cbar=False
    )

    # surface data
    data = nt.surf_from_cifti(cifti_img)
    lh_data = data[0].squeeze()
    rh_data = data[1].squeeze()

    if threshold is not None:
        lh_data[lh_data < threshold] = np.nan
        rh_data[rh_data < threshold] = np.nan

    if cscale == 'sym':
        maxval = max(np.nanmax(np.abs(lh_data)), np.nanmax(np.abs(rh_data)))
        cscale = [-maxval, maxval]
    elif cscale == 'from0':
        maxval = max(np.nanmax(lh_data), np.nanmax(rh_data))
        cscale = [0, maxval]

    p.add_layer(
        {'left': lh_data, 'right': rh_data},
        cmap=cmap,
        color_range=cscale,
        alpha=alpha,
        zero_transparent=zero_transparent,
        cbar=cbar
    )

    fig = p.build(figsize=figsize)
    fig.show()