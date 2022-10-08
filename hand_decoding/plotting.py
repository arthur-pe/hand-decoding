from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from functions import *
from analysis_functions import *
import matplotlib.patches as mpatches

"""
Throughout:
ax (matplotlib ax): an axis on which to plot
"""

def plot_hand_movement(ax, movement, angle, linestyle='-', alpha=1.0, linewidth=1.0, set_ax_lim=True):
    """
    Plots hand movement with trajectories colored as angle
    in:
        movement (ndarray): time x trial x XY
        angle (ndarray): len(angle) = trial, angle must be between 0 and 2pi
    """

    cmap = matplotlib.cm.get_cmap('gist_rainbow')

    for i in range(movement.shape[1]):
        ax.plot(movement[:, i, 0], movement[:, i, 1], color=cmap(angle[i]/(2*np.pi)),
                linestyle=linestyle, alpha=alpha, linewidth=linewidth)

    max_movement = np.abs(movement).max()

    if set_ax_lim:
        ax.set_xlim(-max_movement*1.05, max_movement*1.05)
        ax.set_ylim(-max_movement*1.05, max_movement*1.05)

    ax.set_aspect('equal')

def plot_multiple_hand_movements(axs, movements, angles, linestyle='-', alpha=1.0, linewidth=1.0):
    """
    Plots hand movement with trajectories colored as angle, puts in separate axes
    in:
        movement (ndarray): time x trial x XY
        angle (ndarray): len(angle) = trial, angle must be between 0 and 2pi
    """

    max_x = np.max(np.array([np.abs(m[...,0]).max() for m in movements]))
    max_y = np.max(np.array([np.abs(m[...,1]).max() for m in movements]))

    max_movement = max((max_x, max_y))

    for i in range(len(movements)):
        plot_hand_movement(axs[i], movements[i], angles[i],
                linestyle=linestyle, alpha=alpha, linewidth=linewidth)
        axs[i].set_xlim(-max_movement, max_movement)
        axs[i].set_ylim(-max_movement, max_movement)

def plot_lda_pos(ax, decoded_initial_state, condition, angle, label=False):
    """
    Plots decoded initial state
    in:
        decoded_init_state (ndarray): trial x 3. The first dimension is LDA decoded condition and the other two the target pos
        condition (ndarray): trial. The condition decoded by LDA
        angle (ndarray): trial. The angle of the decoded target.
    """

    cmap = matplotlib.cm.get_cmap('gist_rainbow')

    z = -decoded_initial_state.std()

    for mi, m in enumerate(decoded_initial_state):
        if condition[mi] == 0:
            marker = '^'
        else:
            marker = 'o'
        ax.plot(m[0], m[1], m[2],
                marker, color=cmap(angle[mi]), alpha=1, ms=5)
        ax.plot(m[0], m[1], z, marker, color=cmap(angle[mi]), alpha=.2, ms=5)

    #ax.set_xlim(-.035, .035)
    #ax.set_ylim(-.06, .06)
    #ax.set_zlim(-.06, .04)
    ax.set_zlim(z,decoded_initial_state[...,2].max())

    ax.set_xlabel('condition 1 vs. 2')
    ax.set_ylabel('target dim. 1')
    ax.set_zlabel('target dim. 2')

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((0.95, 0.95, 0.95, 1.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

    if not label:
        ax.set_xticks([],[])
        ax.set_yticks([],[])
        ax.set_zticks([],[])

    # plt.axis('off')
    ax.view_init(elev=14, azim=-114)

def plot_bar_per_condition(ax, value, error=None, label_model=None, label_condition=None):
    """
    Plots bar per condition
    int:
        values (ndarray): #model x #condition. model is the across, condition is the within.
    """

    number_models, number_conditions = value.shape

    cmap = matplotlib.cm.get_cmap('pink')

    for i in range(0,number_models):
        x = np.arange(i*(number_conditions+1), (i+1)*(number_conditions+1),1)
        y = value[i]
        if error is not None:
            ax.bar(x[:-1], y, yerr=error[i], color=[cmap(j/number_conditions) for j in range(number_conditions)], align='center')
        else:
            ax.bar(x[:-1], y, color='grey')

    patches = [mpatches.Patch(color=cmap(j/number_conditions), label=label_condition[j]) for j in range(number_conditions)]
    ax.legend(handles=patches)

    if label_model is not None:
        ax.set_xticks(np.arange(0, (number_models)*(number_conditions+1),number_conditions+1)+number_conditions/2-0.5, label_model)

def plot_correlation_matrix(ax, fig, correlation_matrix, cuts=None):
    """
    Plots a correlation matrix.
    in:
        fig (matplotlib fig): a figure to add the color bar to the axis
        correlation_matrix: a correlation matrix
    """

    im = ax.imshow(correlation_matrix, cmap='gnuplot2', vmin=-.2, vmax=.2, origin='upper', aspect='equal')
    if cuts is not None:
        ax.set_xticks(cuts)
        ax.set_yticks(cuts)

    fig.colorbar(im, shrink=.6, label='Correlation', ticks=[-.2, 0, .2])

if __name__=='__main__':
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot()
    correlation_matrix = np.corrcoef(np.random.randn(100,50))
    cuts = np.array([10*i for i in range(1,11)])
    plot_correlation_matrix(ax, fig, correlation_matrix, cuts)
    plt.show()

    """fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot()
    y = np.array([[1,2,3,4],[2,3,3,4],[2,1,3,4],[2,1,3,4],[2,1,3,4]])
    err = np.ones_like(y)/4
    labels = ['raw', 'tca', 'pca', 'slicetca1', 'slicetca2']
    plot_bar_per_condition(ax, y, err, labels, label_condition=['a', 'b', 'c', 'd', 'e'])
    plt.show()"""

    """movement = ((np.random.rand(200, 2)[:, :, np.newaxis] * 2 - 1) @ np.linspace(0, 1, 101)[np.newaxis, :]).transpose(2, 0,1)
    angle = (np.arctan2(movement[1, :, 1], movement[1, :, 0]) + np.pi) / (2 * np.pi)

    condition = (np.random.rand(200)<0.5)#.astype(int)

    neural_activity = movement @ np.random.randn(2,50)

    neural_activity[:, condition] += 0.1
    condition = condition.astype(int)

    print(movement.shape, neural_activity.shape, condition.shape, angle.shape)

    decoded_initial_state = neural_activity[1] @ lda_pos(neural_activity[1], condition, movement[-1]).T

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(projection='3d')
    plot_lda_pos(ax, decoded_initial_state, condition, angle)

    plt.show()"""

    """movement = ((np.random.rand(200,2)[:,:,np.newaxis]*2-1) @ np.linspace(0,1,101)[np.newaxis, :]).transpose(2,0,1)
    angle = (np.arctan2(movement[1,:,1], movement[1,:,0])+np.pi)/(2*np.pi)

    movement2 = ((np.random.rand(200,2)[:,:,np.newaxis]*2-1) @ np.linspace(0,1,101)[np.newaxis, :]).transpose(2,0,1)
    angle2 = (np.arctan2(movement2[1,:,1], movement2[1,:,0])+np.pi)/(2*np.pi)

    movements, angles = [movement, movement2], [angle, angle2]

    plot_multiple_hand_movements(movements, angles)

    plt.show()"""

    """print(movement.shape)

    angle = (np.arctan2(movement[1,:,1], movement[1,:,0])+np.pi)/(2*np.pi)

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot()
    plot_hand_movement(ax, movement.transpose(1,0,2), angle)

    plt.show()"""

