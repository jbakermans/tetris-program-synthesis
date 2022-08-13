#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 12:08:44 2022

@author: jbakermans
"""

import dsl
import pcfg
import matplotlib.pyplot as plt

# Create pcfg
grammar = pcfg.GRAMMAR

# Sample a bunch of programs
programs = [pcfg.sample(grammar) for _ in range(16)]

# Build shapes for programs
shapes = [dsl.run_program(p) for p in programs]

# Plot results
plt.figure();
for r in range(4):
    for c in range(4):
        # Calculate index for current program
        i = r * 4 + c
        # Create subplot
        ax = plt.subplot(4, 4, i + 1)
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])    
        # Plot shape
        if shapes[i] is not None:
            ax.imshow(shapes[i], cmap='Greys', vmin=0, vmax=1)
        # Set title to program
        ax.set_title(programs[i])


