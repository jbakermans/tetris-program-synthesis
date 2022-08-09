#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 10:07:21 2022

@author: jbakermans
"""

import numpy as np

# Create primitives as global
PRIMITIVES = (
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
    # If the shift is so big that b will miss to a: break immediately
    if shift <= -b.shape[0] or shift >= a.shape[0]:
        return None
    # Place object a at origin
    pos_a = [0, 0]
    # Squeeze object b towards object a until they overlap
    squeeze = 0
    while not do_overlap([a, b], [pos_a, [shift, a.shape[1] - (squeeze + 1)]]):
        squeeze += 1
    # Place object b next to object a, shifted vertically, squeezed horizontally
    pos_b = [shift, a.shape[1] - squeeze]
    # Return shape that combines the two
    return combine([a, b], [pos_a, pos_b])

# Define horizontal concatenation function
def vert(a, b, shift=0):
    # If the shift is so big that b will miss a: break immediately
    if shift <= -b.shape[1] or shift >= a.shape[1]:
        return None
    # Place object a at origin
    pos_a = [0, 0]
    # Squeeze object b towards object a until they overlap
    squeeze = 0
    while not do_overlap([a, b], [pos_a, [a.shape[0] - (squeeze + 1), shift]]):
        squeeze += 1
    # Place object b next to object a, shifted vertically, squeezed horizontally
    pos_b = [a.shape[0] - squeeze, shift]
    # Return shape that combines the two
    return combine([a, b], [pos_a, pos_b])
    
# Combine two shapes at given positions to new shape
def combine(shapes, positions):
    # Find coordinates of canvas that tightly fits the combined shape
    canvas_r = min([p[0] for p in positions])
    canvas_c = min([p[1] for p in positions])
    canvas_h = max([p[0] - canvas_r + s.shape[0] for s, p in zip(shapes, positions)])
    canvas_w = max([p[1] - canvas_c + s.shape[1] for s, p in zip(shapes, positions)])
    # Create empty canvas
    canvas = np.zeros([canvas_h, canvas_w])
    # Add shapes to canvas
    for s, p in zip(shapes, positions):
        # Find canvas locations to draw: positions, with canvas offset added
        r, c = p[0] - canvas_r, p[1] - canvas_c
        # Select part of canvas draw on
        draw_area = canvas[r:(r + s.shape[0]), c:(c + s.shape[1])]
        # And colour in all values that are 1 in the shape
        draw_area[s == 1] = 1
    # Return canvas with input shapes on it
    return canvas

# Find if given shapes at given positions have any overlap
def do_overlap(shapes, positions):
    # Calculate areas of separate and combined shapes
    separate_area = sum([np.sum(s) for s in shapes])
    combined_area = np.sum(combine(shapes, positions))
    # There is overlap if the separate area is larger than the combined area
    return combined_area < separate_area

# Create 'generative model' that enumerates all possible two-shape combinations
def generate(primitives=PRIMITIVES):
    # Build a long list of function-shape pairs
    all_shapes = []
    # First add primitives to list
    for i, s in enumerate(primitives):
        all_shapes.append(['s:' + str(i), s])
    # Then add all pairs of primitives
    for i1, s1 in enumerate(primitives):
        for i2, s2 in enumerate(primitives):
            for nf, f in enumerate([hor, vert]):
                for shift in range(-2, 3):
                    all_shapes.append([
                        'f:' + str(nf) + '_s1:' + str(i1) + '_s2:' + str(i2) + '_v:' + str(shift),
                        f(s1, s2, shift=shift)])
            print('Finished object pair (' + str(i1) + ', ' + str(i2) + ')')
    return all_shapes
    
