import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_heatmaps(df, indices, savepath, interpolation='none', cmap='Spectral',
                  titles=['Heatmap Plot'] * 2, x_label='x', y_label='y',
                  cbar_labels=['Metric'] * 2, is_show=False):
    data_0 = df.pivot(index=indices['y'], columns=indices['x'], values=indices['z0'])
    data_1 = df.pivot(index=indices['y'], columns=indices['x'], values=indices['z1'])

    x_values = data_0.columns
    y_values = data_0.index

    x_ticks = np.linspace(0, len(x_values) - 1, 5, dtype=int)
    x_tick_labels = np.round(x_values[x_ticks], 3)

    y_ticks = np.linspace(0, len(y_values) - 1, 5, dtype=int)
    y_tick_labels = np.round(y_values[y_ticks], 3)

    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    im0 = ax0.imshow(data_0, aspect='auto', origin='lower', cmap=cmap, interpolation=interpolation)
    ax0.set_title(titles[0])
    ax0.set_xlabel(x_label)
    ax0.set_ylabel(y_label)
    ax0.set_xticks(x_ticks)
    ax0.set_xticklabels(x_tick_labels)
    ax0.set_yticks(y_ticks)
    ax0.set_yticklabels(y_tick_labels)
    plt.colorbar(im0, ax=ax0, label=cbar_labels[0])

    im1 = ax1.imshow(data_1, aspect='auto', origin='lower', cmap=cmap, interpolation=interpolation)
    ax1.set_title(titles[1])
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_tick_labels)
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_tick_labels)
    plt.colorbar(im1, ax=ax1, label=cbar_labels[1])

    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    if is_show:
        plt.show()


def plot_pareto(df, savepath, plotcols=['x', 'y', 'pareto'], bg_col='white',
                pareto_col='red', nonpareto_col='blue', labels=['Non-Pareto', 'Pareto'],
                title='Scatter Plot', x_label='x', y_label='y', is_show=False):
    plotdf = df[plotcols]

    plt.figure(figsize=(6, 4))
    plt.gca().set_facecolor(bg_col)
    plt.gca().set_axisbelow(True)

    sns.scatterplot(x=plotcols[0], y=plotcols[1], data=plotdf[~plotdf[plotcols[2]]],
                    label=labels[0], color=nonpareto_col, alpha=0.8, s=70)
    sns.scatterplot(x=plotcols[0], y=plotcols[1], data=plotdf[plotdf[plotcols[2]]],
                    label=labels[1], color=pareto_col, alpha=0.8, s=120)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)
    if is_show:
        plt.show()


def plot_cm(truth, preds, class_names, percs=True, cmap='Blues',
            title='Confusion Matrix', x_label='Predicted Label',
            y_label='True Label', is_show=False):
    cm = confusion_matrix(truth, preds)
    cm = cm.T
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(6, 4))

    if percs:
        sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap=cmap, xticklabels=class_names,
                    annot_kws={"size": 8}, yticklabels=class_names,
                    cbar_kws={'label': 'Percentage (%)'})
    else:
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=class_names,
                    annot_kws={"size": 8}, yticklabels=class_names)

    plt.xticks(fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.title(title, fontsize=12, pad=10)
    plt.xlabel(x_label, fontsize=12, labelpad=10)
    plt.ylabel(y_label, fontsize=12, labelpad=10)
    if is_show:
        plt.show()
