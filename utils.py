#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 21:41:17 2022

@author: jbakermans
"""

import numpy as np
import matplotlib.pyplot as plt

# Define all primitive shapes
primitives = (
    np.array([[1,1],[1,0]]),
    np.array([[1,1],[0,1]]),
    np.array([[1,0],[1,1]]),
    np.array([[0,1],[1,1]]),
    np.array([[1]]),
    np.array([[1],[1]]),
    np.array([[1,1]]),
    np.array([[1],[1],[1]]),
    np.array([[1,1,1]])
    )

# Set canvas side lengths
canvas_l = 8

# Render list of shapes and positions to canvas
def render_shapes(shapes, positions, 
                  canvas=None, primitives=primitives, l=canvas_l):
    # Create empty canvas, if not provided
    canvas = np.zeros([canvas_l, canvas_l]) if canvas is None else canvas
    # shapes is list of primitive indices, positions is list of where they go
    for s, p in zip(shapes, positions):
        # Cut off shape if it doesn't fit
        w = min([primitives[s].shape[1], l - p[0]])
        h = min([primitives[s].shape[0], l - p[1]]) 
        # Copy shape values to appropriate canvast pixels
        canvas[p[0]:(p[0] + h), p[1]:(p[1] + w)] = \
            primitives[s][:h, :w]
    # Return canvas with primitives on it
    return canvas

# Plot canvas
def plot_canvas(canvas, ax=None):
    # Create axes if not existing
    if ax is None: 
        # New axes
        ax = plt.axes()
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
    # Plot input canvas
    ax.imshow(canvas, cmap='Greys')
    # Return axes for further processing
    return ax

# Create shape and plot
shapes = [2, 3, 0, 8]
pos = [[0, 0], [0, 2], [2, 2], [4, 2]]
plot_canvas(render_shapes(shapes, pos))
