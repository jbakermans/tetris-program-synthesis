#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 21:23:47 2022

@author: jbakermans
"""

import json
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import kendalltau

import dsl
import tetris
  
def get_in_solution(solution, target):
    # Find which objects are included in the solution
    in_solution = []
    for i, (pos, shape) in enumerate(zip(data['trial_solution'][trial], dsl.PRIMITIVES)):
        if (pos[0] + shape.shape[1] > 0 and pos[0] < len(data['trial_target'][trial][0])
            and pos[1] + shape.shape[0] > 0 and pos[1] < len(data['trial_target'][trial])):
            in_solution.append(i)
    # Return list of objects in solution, in order of object id
    return in_solution
        
def get_final_moves(in_solution, events):
    # Collect final move for all objects that are part of solution
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
    # Return final moves in order of appearance
    return sorted(final_move, key=lambda d: d['t'])
    
def parse_solution(final_move, plot_pp=False):
    # Run through all final moves, and find if they instantiate any relation
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
            # Mostly for debugging: plot partial programs 
            if plot_pp:
                # Build the shape that the participant has currently built
                progress = dsl.combine(
                    [dsl.PRIMITIVES[m['id']] for m in final_move[:(m_i+1)]],
                    [[m['y'], m['x']] for m in final_move[:(m_i+1)]])
                # Plot current solution and final target on first row
                plt.figure(); plt.subplot(2,2,1); plt.imshow(progress); plt.subplot(2,2,2); plt.imshow(data['trial_target'][trial])
                # Then plot all partial programs that are currently possible
                for i in range(len(pp)):
                    plt.subplot(2,len(pp),len(pp) + i + 1); plt.imshow(pp[i]['shape'])
    # Clean up programs: any partial programs that didn't result in the full one
    pp = [p for p in pp if np.array_equal(p['shape'], data['trial_result'][trial])]
    # Return final partial problems
    return pp


def rank_corrs(order_shape, correct_shape):
    # Get number of shapes and number of participants
    n_participants = len(order_shape)
    n_shapes =  len(order_shape[0])
    # Create a big matrix of N * (N-1) / 2 pairs of kendall tau correlations for each shape
    corrs = np.full((int(n_participants * (n_participants - 1) / 2), n_shapes), np.nan)
    # For each shape: run through all correlations
    for curr_shape in range(n_shapes):
        c = -1
        for i in range(n_participants):
            for j in range(i + 1, n_participants):
                c += 1
                # Only continue if both were correct
                if correct_shape[i][curr_shape] and correct_shape[j][curr_shape]:
                    # Collect sequences to compare
                    seq1 = order_shape[i][curr_shape]
                    seq2 = order_shape[j][curr_shape]
                    # Remove anything that they don't have both
                    seq1 = [s for s in seq1 if s in seq2]
                    seq2 = [s for s in seq2 if s in seq1]
                    # Set sequence 1 to the index of appearance in 2
                    seq1 = [seq2.index(s) for s in seq1]
                    # And then sequence 2 is simply in order
                    seq2 = [i for i in range(len(seq2))]
                    # Then calculate kendall tau rank correlation
                    tau, _ = kendalltau(seq1, seq2)
                    # Only add if finite
                    if np.isfinite(tau): corrs[c, curr_shape] = tau
    return corrs
    

# List folders in data directory
participants = sorted([i for i in os.listdir('./data') if '.txt' in i])
# Return root for directory of selected session
all_files = [os.path.join('./data', p) for p in participants]
# Load all of them
all_data = []
for filename in all_files:
    # Open data file
    f = open(filename)
    # Parse string to dictionary
    data = json.load(f)
    # Close file
    f.close()
    # An append data
    all_data.append(data)

# Collect size of dataset
n_participants = len(all_data)
n_trials = all_data[0]['n_trials']

# Create data matrices, with each column a trial
correct_trial = np.zeros((n_participants, n_trials))
time_trial = np.zeros((n_participants, n_trials))
primitives_trial = np.zeros((n_participants, n_trials))
moves_trial = np.zeros((n_participants, n_trials))
# Afterwards reshuffle columns for each participant so they represent shapes
correct_shape = np.zeros((n_participants, n_trials))
time_shape = np.zeros((n_participants, n_trials))
primitives_shape = np.zeros((n_participants, n_trials))
moves_shape = np.zeros((n_participants, n_trials))
# Collect order of primitives in each shape
order_shape = [[[] for _ in range(n_trials)] for _ in range(n_participants)]

# Run through trials
for d_i, data in enumerate(all_data):
    for trial in range(n_trials):
        # Print progress
        print(f"Processing sub {d_i+1} / {len(all_data)}, trial {trial+1} / {data['n_trials']}...")  
        # Find which primitives ended up in solution
        in_solution = get_in_solution(data['trial_solution'][trial], data['trial_target'][trial])
        # Find final moves: the shapes in the solution dragged to their final position
        final_moves = get_final_moves(in_solution, data['trial_events'][trial])
        # Collect values
        correct_trial[d_i, trial] = data['trial_correct'][trial]
        time_trial[d_i, trial] = data['trial_time'][trial]/1000
        primitives_trial[d_i, trial] = len(in_solution) if correct_trial[d_i, trial] else np.nan
        moves_trial[d_i, trial] = len(data['trial_events'][trial])
        # Store order
        order_shape[d_i][data['trial_target_ids'][trial]] = [m['id'] for m in final_moves]
        
# Reorder data for shapes rather than trials
for d_i, data in enumerate(all_data):
    # Store the same values in different locations, for target id
    target_ids = data['trial_target_ids']
    correct_shape[d_i, target_ids] = correct_trial[d_i]
    time_shape[d_i, target_ids] = time_trial[d_i]
    primitives_shape[d_i, target_ids] = primitives_trial[d_i]
    moves_shape[d_i, target_ids] = moves_trial[d_i]
    
# Z-score times
time_shape_z = (time_shape - np.nanmean(time_shape, axis=1).reshape(-1,1)) \
    / np.nanstd(time_shape, axis=1).reshape(-1,1)
# Find order of shapes by z-scored time: measure of difficulty
target_ids_sorted = np.argsort(np.nanmean(time_shape_z, axis=0))
# Get all targets by 'un-randomising' the target order for one participant
targets = [[] for _ in range(n_trials)]
for trial in range(n_trials):
    targets[all_data[5]['trial_target_ids'][trial]] = all_data[5]['trial_target'][trial]
# Then sort the targets according to 
targets_sorted = [targets[i] for i in target_ids_sorted]

# Now re-order everything with shapes according to difficulty
correct_shape = correct_trial[:, target_ids_sorted]
time_shape = time_trial[:, target_ids_sorted]
primitives_shape = primitives_trial[:, target_ids_sorted]
moves_shape = moves_trial[:, target_ids_sorted]
time_shape_z = time_shape_z[:, target_ids_sorted]
order_shape = [[curr_order[i] for i in target_ids_sorted] for curr_order in order_shape]

# Calculate rank correlations
corr_tau = rank_corrs(order_shape, correct_shape)
# Convert Kendall tau to Pearson r 
# See https://www.cedu.niu.edu/~walker/personal/Walker%20Kendall's%20Tau.pdf
corr_r = np.sin(0.5 * np.pi * corr_tau)
# Convert Pearson r to Fisher z
corr_z = np.arctanh(corr_r)

# Plot trials
plt.figure()
for r_i, (y, y_label) in enumerate(zip(
        [correct_trial, time_trial, primitives_trial, moves_trial],
        ['Correct (T/F)', 'Time (s)', '# Primitives (1)', '# Moves (1)'])):
        plt.subplot(4, 1, r_i + 1)
        plt.plot(np.arange(y.shape[1]), y.transpose(), color=(0.8, 0.8, 0.8))
        plt.errorbar(np.arange(y.shape[1]), np.nanmean(y, axis=0), np.nanstd(y, axis=0)/np.sqrt(y.shape[1]), color=(0,0,0))
        plt.ylabel(y_label)
        #plt.title(names)
        if r_i == 3:
            plt.xlabel('Trials')
            
# Plot stimuli            
plt.figure()
for r_i, (y, y_label) in enumerate(zip(
        [correct_shape, time_shape_z, primitives_shape, moves_shape],
        ['Correct (T/F)', 'Time (z-scored)', '# Primitives (1)', '# Moves (1)'])):
        plt.subplot(4, 1, r_i + 1)
        plt.plot(np.arange(y.shape[1]), y.transpose(), color=(0.8, 0.8, 0.8))
        plt.errorbar(np.arange(y.shape[1]), np.nanmean(y, axis=0), np.nanstd(y, axis=0)/np.sqrt(y.shape[1]), color=(0,0,0))
        plt.ylabel(y_label)
        #plt.title(names)
        if r_i == 3:
            plt.xlabel('Stimuli')            

# Plot stimuli, but only the interesting panels
plt.figure()
plt.subplot(2,1,1)
y = time_shape_z
plt.plot(np.arange(y.shape[1]), y.transpose(), color=(0.8, 0.8, 0.8))
plt.errorbar(np.arange(y.shape[1]), np.nanmean(y, axis=0), np.nanstd(y, axis=0)/np.sqrt(y.shape[1]), color=(0,0,0))
plt.ylabel('Time (z-scored)')
plt.subplot(2,1,2)
plt.violinplot(dataset=[corr_tau[np.isfinite(corr_tau[:, i]), i] for i in range(n_trials)],
               positions=range(n_trials),
               widths=0.9)
plt.ylabel('All pair Kendall tau rank correlation')
plt.xlabel('Stimuli')

# Plot stimulus shapes from easy to hard            
plt.figure()
s_to_plot = [np.array(targets_sorted[16])] + [dsl.PRIMITIVES[i] for i in [2, 3, 0, 1, 6]]
for i, s_i in enumerate(s_to_plot):
    plt.subplot(len(s_to_plot), 1, i + 1)
    plt.imshow(s_i, cmap='Greys', vmin=0, vmax=1)
    plt.axis('off')
    if i == 0:
        plt.title('Stimulus 16')
    else:
        plt.title('Block # ' + str(i)) 
                
# Plot stimulus shapes from easy to hard            
plt.figure()
s_to_plot = [i for i in range(0, len(targets_sorted), 6)]
if s_to_plot[-1] < (len(targets_sorted) - 1): s_to_plot.append(len(targets_sorted) - 1)
for i, s_i in enumerate(s_to_plot):
    plt.subplot(len(s_to_plot), 1, i + 1)
    plt.imshow(targets_sorted[s_i], cmap='Greys', vmin=0, vmax=1)
    plt.axis('off')
    plt.title('Stimulus ' + str(s_i + 1))    
    
    
