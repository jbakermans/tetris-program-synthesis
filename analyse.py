#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 21:23:47 2022

@author: jbakermans
"""

import json
import dsl
import tetris
import numpy as np
from matplotlib import pyplot as plt
  
# Open data file
f = open('./data/data_v1_2022-08-18_21-21-01.txt')
# Parse string to dictionary
data = json.load(f)
# Close file
f.close()

# I want to collect all possible parses for each solution
programs = {'in_solution': [], 'final_move': [], 'program': []}
# Run through trials
for trial in range(data['n_trials']):
    # Print progress
    print(f"Processing trial {trial} / {data['n_trials']}...")    
    # Only include correct trials
    if data['trial_correct'][trial]:
        # Find which objects are included in the solution
        in_solution = []
        for i, (pos, shape) in enumerate(zip(data['trial_solution'][trial], dsl.PRIMITIVES)):
            if (pos[0] + shape.shape[1] > 0 and pos[0] < len(data['trial_target'][trial][0])
                and pos[1] + shape.shape[0] > 0 and pos[1] < len(data['trial_target'][trial])):
                in_solution.append(i)
        programs['in_solution'].append(in_solution)
        # Find final move for each object in solution
        final_move = []
        for i in in_solution:
            for t, move in enumerate(data['trial_events'][trial][::-1]):
                if move['id'] == i:
                    # Swap x and y 
                    final_move.append({'t': len(data['trial_events'][trial]) - t,
                                      'x': move['release']['x'], 
                                      'y': move['release']['y'],
                                      'id': i})
                    break
        programs['final_move'].append(sorted(final_move, key=lambda d: d['t']))   
        # Now run through all final moves, and find if they instantiate any relation
        pp = []
        for m_i, move in enumerate(final_move):
            # Print progress
            print(f'- Parsing move {str(m_i)} / {str(len(final_move))}...')
            # Translate move into program dictionary
            p_move = {'shape': dsl.PRIMITIVES[move['id']], 
                       'primitives': [{'s': dsl.PRIMITIVES[move['id']], 'r': move['y'], 'c': move['x']}],
                       'program': tetris.Primitive(move['id'])}
            # If there are no partial programs yet: immediately add current program
            if len(pp) < 1: 
                pp.append(p_move)
            else:
                # Add shape for current move to stuff for testing
                to_test = [p_move]
                # Keep track of shapes that have been used in combinations to remove them afterwards
                to_remove = []
                while len(to_test) > 0:
                    # Select first entry from to_test as current move
                    curr_test = to_test.pop(0)
                    # Keep track of each of the new combinations that need to be added
                    to_add =[]                    
                    # Find if the current move is adjacent to any of the existing partial programs
                    for p_i, p in enumerate(pp):
                        # Test if placing current move and existing programs in relation is equal to existing shape
                        for f in [dsl.hor, dsl.vert]:
                            for order in [0, 1]:
                                for shift in range(-tetris.RESOLUTION,tetris.RESOLUTION):
                                    # Get inputs for function: current object and other object
                                    f_in = [curr_test['shape'],
                                            p['shape']]
                                    # Flip around for order equals 1
                                    if order == 1:
                                        f_in = f_in[::-1]
                                    # What does current function, order, shift with current pp produce?
                                    new_combination = f(f_in[0], f_in[1], shift)
                                    # What do you get when you combine solution shapes?
                                    user_combination = dsl.combine(
                                        [curr_p['s'] for curr_p in curr_test['primitives']] + [curr_p['s'] for curr_p in p['primitives']],
                                        [[curr_p['r'], curr_p['c']] for curr_p in curr_test['primitives']] + [[curr_p['r'], curr_p['c']] for curr_p in p['primitives']])
                                    # Do these two match? If so, found relation!
                                    if np.array_equal(new_combination, user_combination):
                                        # Fine new program
                                        if f == dsl.hor:
                                            if order == 0:
                                                new_program = tetris.Hor(curr_test['program'], 
                                                                         p['program'], shift=shift)
                                            else:
                                                new_program = tetris.Hor(p['program'],
                                                                         curr_test['program'], shift=shift)
                                        else: 
                                            if order == 0:
                                                new_program = tetris.Vert(curr_test['program'], 
                                                                         p['program'], shift=shift)
                                            else:
                                                new_program = tetris.Vert(p['program'],
                                                                         curr_test['program'], shift=shift)
                                        # Remove old program
                                        to_remove.append(p_i)                                                
                                        # Add new combined program
                                        to_add.append({'shape': new_combination, 
                                                       'primitives': p['primitives'] + curr_test['primitives'],
                                                       'program': new_program})
                                        # And add the new combined program to stuff that needs to be tested
                                        to_test.append({'shape': new_combination, 
                                                       'primitives': p['primitives'] + curr_test['primitives'],
                                                       'program': new_program})
                    # Add new combination to list of partial programs
                    pp += to_add
                # At the very end of this move: remove any partial programs that have now been included in larger ones
                if len(to_remove) == 0:
                    # If nothing was combined into bigger things: current move is a new primitive
                    pp.append(p_move)
                else:
                    # Remove the partial programs as necessary
                    pp = [p for p_i, p in enumerate(pp) if p_i not in to_remove]
            # if trial == 1:
            #     progress = dsl.combine(
            #         [dsl.PRIMITIVES[m['id']] for m in final_move[:(m_i+1)]],
            #         [[m['y'], m['x']] for m in final_move[:(m_i+1)]])
            #     plt.figure(); plt.subplot(2,2,1); plt.imshow(progress); plt.subplot(2,2,2); plt.imshow(data['trial_target'][trial])
            #     for i in range(len(pp)):
            #         plt.subplot(2,len(pp),len(pp) + i + 1); plt.imshow(pp[i]['shape'])
            #     import pdb; pdb.set_trace()
        # Clean up programs: any partial programs that didn't result in the full one
        pp = [p for p in pp if np.array_equal(p['shape'], data['trial_result'][trial])]
        # Add pp to programs
        programs['program'].append(pp)
                    
# Plot results
for p in programs['program']:
    plt.figure();
    for i, pp in enumerate(p):
        plt.subplot(len(p), 1, i + 1)
        plt.imshow(pp['shape'])
        plt.title(pp['program'])        