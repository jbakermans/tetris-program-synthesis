#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 10:07:21 2022

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

# Define horizontal concatenation function
def hor(a, b, shift=0):
    # Assume object a is placed at origin
    pos_a = [0, 0]
    # Place object b to the right of a's bounding box, shifted vertically by shift
    pos_b = [shift, a.shape[1]]
    # Return shape that combines the two
    return combine([a, b], [pos_a, pos_b])

# Define vertical concatenation function
def vert(a, b, shift=0):
    # Assume object a is placed at origin
    pos_a = [0, 0]
    # Place object b to the bottom of a's bounding box, shifted horizontally by shift
    pos_b = [a.shape[0], shift]
    # Return shape that combines the two
    return combine([a, b], [pos_a, pos_b])
    
# Combine two shapes at given positions to new shape
def combine(shapes, positions):
    # Find coordinates of canvas that tightly fits the combined shape
    canvas_r = min([p[0] for p in positions])
    canvas_c = min([p[1] for p in positions])
    canvas_h = max([p[0] + s.shape[0] for s, p in zip(shapes, positions)])
    canvas_w = max([p[1] + s.shape[1] for s, p in zip(shapes, positions)])
    # Create empty canvas
    canvas = np.zeros([canvas_h, canvas_w])
    # Add shapes to canvas
    for s, p in zip(shapes, positions):
        # Find canvas locations to draw: positions, with canvas offset added
        r, c = p[0] - canvas_r, p[1] - canvas_c
        # Copy shape values to appropriate canvast pixels
        canvas[r:(r + s.shape[0]), c:(c + s.shape[1])] = s
    # Return canvas with input shapes on it
    return canvas

# Build a complex shape by combining simple shapes
x1 = hor(primitives[2], primitives[3])
x2 = vert(primitives[0], primitives[8])
x3 = vert(x1, x2, shift=2)

# Plot all shapes
plt.figure(); 
# First row: primitive shapes
for i, s in enumerate(primitives):
    # Create subplot
    ax = plt.subplot(2, len(primitives), i + 1)
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])    
    # Plot shape
    ax.imshow(s, cmap='Greys', vmin=0, vmax=1)
    # Set title
    ax.set_title('Primitive shape ' + str(i))
# Second row: combined shapes
for i, (s, n) in enumerate(zip(
        [x1, x2, x3], ['hor(2, 3)', 'vert(0, 8)', 'vert(hor(2, 3), vert(0, 8))'])):
    # Create subplot
    ax = plt.subplot(2, 3, 3 + i + 1)
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])    
    # Plot shape
    ax.imshow(s, cmap='Greys', vmin=0, vmax=1)
    # Set title
    ax.set_title(n)    
        
