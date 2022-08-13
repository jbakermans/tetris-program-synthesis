#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 16:40:01 2022

@author: jbakermans
"""

import numpy as np
from nltk import PCFG
from nltk.grammar import Nonterminal, ProbabilisticProduction

# Define the grammar
GRAMMAR = PCFG.fromstring("""
S -> p [1.0]
p -> s [0.6] | hor [0.2] | vert [0.2]
s -> 's0' [0.12] | 's1' [0.11] | 's2' [0.11] | 's3' [0.11] | 's4' [0.11] | 's5' [0.11] | 's6' [0.11] | 's7' [0.11] | 's8' [0.11]
hor -> 'hor(' p ',' p ';' i ')' [1.0]
vert -> 'vert(' p ',' p ';' i ')' [1.0]
i -> '-3' [0.05] | '-2' [0.05] | '-1' [0.15] | '0' [0.5] | '1' [0.15] | '2' [0.05] | '3' [0.05]
""")

# Create a lambda function for recursively flattening lists
flatten=lambda l: sum(map(flatten,l),[]) if isinstance(l,list) else [l]

# Sample a sentence from a probabilitistic grammar
def sample(grammar):
    # Sample from start, flatten result, and convert to program string
    return ''.join(flatten(sample_rule(grammar, grammar.start())))

# Sample productions from a rule, recursively
def sample_rule(grammar, lhs):
    # If the lhs is non-terminal: sample from all its productions
    if isinstance(lhs, Nonterminal):
        # Get all productions with given lhs and their probabilities
        productions = grammar.productions(lhs=lhs)
        # Sample which production to use
        production = np.random.choice(productions, p=[p.prob() for p in productions])
        # Repeat (recursion!) sampling for each production
        return [sample_rule(grammar, lhs) for lhs in production.rhs()]
    else:
        # For terminal left hand side: simply return the lhs itself
        return lhs

# Get all probabilities from grammar
def get_probabilities(grammar):
    # Collect rules: all unique lhs in production
    rules = np.unique([p.lhs() for p in grammar.productions()])
    # For each lhs: get probabilities of all possible rhs
    return [[rhs.prob() for rhs in grammar.productions(lhs=lhs)] for lhs in rules]

# Set all probabilites in grammar
def set_probabilities(grammar, probabilities):
    # Probabilities are immutable, so create a new grammar with updated probabilities
    new_start = grammar.start()
    new_productions = []
    # Collect rules: all unique lhs in production
    rules = np.unique([p.lhs() for p in grammar.productions()])
    # Copy lhs, rhs from grammar probabilities then set probability
    for lhs, ps in zip(rules, probabilities):
        # Normalise probabilities
        ps = [p / sum(ps) for p in ps]
        # Create new probabilistic productions with those probabilities
        for rhs, p in zip([p.rhs() for p in grammar.productions(lhs=lhs)], ps):
            new_productions.append(ProbabilisticProduction(lhs, rhs, prob=p))
    # Then build a new grammar from new start and new productions
    return PCFG(new_start, new_productions)