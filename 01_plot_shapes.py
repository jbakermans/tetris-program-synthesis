#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 16:19:00 2022

@author: jbakermans
"""

import dsl
import matplotlib.pyplot as plt
import random

# Define all primitive shapes
primitives = dsl.PRIMITIVES

# Build a complex shape by combining simple shapes
x1 = dsl.hor(primitives[2], primitives[3])
x2 = dsl.vert(primitives[0], primitives[8])
x3 = dsl.vert(x1, x2, shift=2)

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
        
# Plot a few squeezed shapes
x4 = dsl.hor(primitives[2], primitives[3], shift=-1)
x5 = dsl.hor(dsl.hor(dsl.vert(primitives[7], primitives[7]), primitives[8], shift=3), primitives[4])
x6 = dsl.hor(dsl.hor(dsl.vert(primitives[7], primitives[7]), primitives[8], shift=3), primitives[4], shift=3)
plt.figure()
for i, s in enumerate([x4, x5, x6]):
    # Create subplot
    ax = plt.subplot(1, 3, i + 1)
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])    
    # Plot shape
    ax.imshow(s, cmap='Greys', vmin=0, vmax=1)
    
# Get all possible two-object combinations, and plot a random sample
all_shapes = dsl.generate()
plot_shapes = random.sample([s for s in all_shapes if s[1] is not None], 25)
plt.figure()
for i, s in enumerate(plot_shapes):
    # Create subplot
    ax = plt.subplot(5, 5, i + 1)
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])    
    # Plot shape
    if s[1] is not None: ax.imshow(s[1], cmap='Greys', vmin=0, vmax=1)
    # Set title
    ax.set_title(s[0])