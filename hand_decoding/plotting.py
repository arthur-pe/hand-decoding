from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from .functions import *

"""
Throughout:
ax (matplotlib ax): an axis on which to plot
"""

def plot_hand_movement(ax, movement, angle, linestyle='-', alpha=1.0, linewidth=1.0, set_ax_lim=True):
    """
    Plots hand movement with trajectories colored as angle
    in:
        ax
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

    print(max_movement)

    for i in range(len(movements)):
        plot_hand_movement(axs[i], movements[i], angles[i],
                linestyle=linestyle, alpha=alpha, linewidth=linewidth)
        axs[i].set_xlim(-max_movement, max_movement)
        axs[i].set_ylim(-max_movement, max_movement)

if __name__=='__main__':

    movement = ((np.random.rand(200,2)[:,:,np.newaxis]*2-1) @ np.linspace(0,1,101)[np.newaxis, :]).transpose(2,0,1)
    angle = (np.arctan2(movement[1,:,1], movement[1,:,0])+np.pi)/(2*np.pi)

    movement2 = ((np.random.rand(200,2)[:,:,np.newaxis]*2-1) @ np.linspace(0,1,101)[np.newaxis, :]).transpose(2,0,1)
    angle2 = (np.arctan2(movement2[1,:,1], movement2[1,:,0])+np.pi)/(2*np.pi)

    movements, angles = [movement, movement2], [angle, angle2]

    plot_multiple_hand_movements(movements, angles)

    plt.show()

    """print(movement.shape)

    angle = (np.arctan2(movement[1,:,1], movement[1,:,0])+np.pi)/(2*np.pi)

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot()
    plot_hand_movement(ax, movement.transpose(1,0,2), angle)

    plt.show()"""

