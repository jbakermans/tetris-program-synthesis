#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 22:22:13 2022

@author: jbakermans
"""

import pickle
import numpy as np

from API import *

from randomSolver import *
from pointerNetwork import *
from programGraph import *
from SMC import *
from ForwardSample import *
from MCTS import MCTS
from CNN import *

import dsl

import time
import random
from scipy.signal import convolve2d

RESOLUTION = 16

import torch
import torch.nn as nn

class Tetris(Program):
    lexicon = ['h', 'v'] \
        + [str(i) for i in range(9)] \
            + [i - int(RESOLUTION / 2) for i in range(RESOLUTION)]

    def __init__(self):
        self._rendering = None

    def __repr__(self):
        return str(self)

    def __ne__(self, o): return not (self == o)

    def execute(self):
        # Only render if this has not been rendered before. Pad to RESOLUTION
        if self._rendering is None: self._rendering = padToFit(self.render())
        return self._rendering

    def IoU(self, other):
        # Calculate 'area under curve' between self and other object
        if isinstance(other, Tetris): other = other.execute()
        # Pad other by own size so they can be convolved
        other = np.pad(other,[[other.shape[0], other.shape[0]],
                              [other.shape[1], other.shape[1]]])
        # Make list of shapes; mirror other for convolution
        shapes = [self.execute(), np.flipud(np.fliplr(other))]
        # Find number of pixels in both self AND other for all shifts
        both = convolve2d(shapes[0], shapes[1])
        # Find number of pixels in either self OR (= 1-AND) other for all shifts
        either = max([np.prod(s.shape) for s in shapes]) \
            - convolve2d(1 - shapes[0], 1 - shapes[1], fillvalue=1)
        # Return the maximum ratio of both over either pixels
        return np.max(both[either>0] / either[either>0])
    
    def render(self):
        # Render function will depend on operation; return None by default
        return None

# The type of CSG's
tTetris = BaseType(Tetris)

class Primitive(Tetris):
    token = 's' # Will be replaced by primitive nr on init
    type = arrow(tTetris, tTetris)
    
    def __init__(self, index=0):
        super().__init__()        
        self.shape = dsl.PRIMITIVES[index]
        self.index = index
        self.token = str(index)
        
    def toTrace(self): return [self]

    def __str__(self):
        return 's' + self.token

    def children(self): return []

    def __eq__(self, o):
        return isinstance(o, Primitive) and o.index == self.index

    def __hash__(self):
        return hash((self.token))
    
    def __contains__(self, p):
        # Use contains to find if queried program occurs anywhere in full program
        return p == self

    def serialize(self):
        return (self.token)
    
    def render(self):
        return self.shape
    
class Hor(Tetris):
    token = 'h'
    type = arrow(integer(-int(RESOLUTION / 2), RESOLUTION - 1 - int(RESOLUTION / 2)), 
                 tTetris, tTetris, tTetris)
    
    def __init__(self, a, b, shift=0):
        super(Hor, self).__init__()
        self.elements = [a,b]
        self.shift = shift

    def toTrace(self):
        return self.elements[0].toTrace() + self.elements[1].toTrace() + [self]

    def __str__(self):
        return f"hor({str(self.elements[0])}, {str(self.elements[1])}; {str(self.shift)})"

    def children(self): return self.elements

    def serialize(self):
        return (self.token,list(self.elements)[0],list(self.elements)[1], self.shift)

    def __eq__(self, o):
        return isinstance(o, Hor) and tuple(o.elements) == tuple(self.elements) \
            and o.shift == self.shift

    def __contains__(self, p):
        # Use contains to find if queried program occurs anywhere in full program
        if p == self: 
            return True
        else: 
            return any(p in e for e in self.elements)

    def __hash__(self):
        return hash((self.token, tuple(self.elements)))
    
    def render(self):
        return dsl.hor(self.elements[0].render(), 
                       self.elements[1].render(), 
                       shift=self.shift)
    
    
class Vert(Tetris):
    token = 'v'
    type = arrow(integer(-int(RESOLUTION / 2), RESOLUTION - 1 - int(RESOLUTION / 2)), 
                 tTetris, tTetris, tTetris)
    
    def __init__(self, a, b, shift=0):
        super(Vert, self).__init__()
        self.elements = [a,b]
        self.shift = shift

    def toTrace(self):
        return self.elements[0].toTrace() + self.elements[1].toTrace() + [self]

    def __str__(self):
        return f"vert({str(self.elements[0])}, {str(self.elements[1])}; {str(self.shift)})"

    def children(self): return self.elements

    def serialize(self):
        return (self.token,list(self.elements)[0],list(self.elements)[1], self.shift)

    def __eq__(self, o):
        return isinstance(o, Vert) and tuple(o.elements) == tuple(self.elements) \
            and o.shift == self.shift
            
    def __contains__(self, p):
        # Use contains to find if queried program occurs anywhere in full program
        if p == self: 
            return True
        else: 
            return any(p in e for e in self.elements)            

    def __hash__(self):
        return hash((self.token, tuple(self.elements)))
    
    def render(self):
        return dsl.vert(self.elements[0].render(), 
                        self.elements[1].render(), 
                        shift=self.shift)    
    
tDSL = DSL([Hor, Vert, Primitive],
          lexicon=Tetris.lexicon)

"""Small utility function"""
def padToFit(dat, val=0, w=RESOLUTION, h=RESOLUTION, center=True):
    # Pad input array with zeros to achieve input shape
    h_add = max([h - dat.shape[0], 0])
    h_add_0 = int(h_add / 2) if center else h_add
    h_add_1 = h_add - h_add_0
    w_add = max([w - dat.shape[1], 0])
    w_add_0 = int(w_add / 2) if center else w_add
    w_add_1 = w_add - w_add_0
    # The resulting padded array has the input roughly in its center
    return np.pad(dat, [[h_add_0, h_add_1],[w_add_0, w_add_1]], 
                  mode='constant', constant_values=val)
    

"""Neural networks"""
class ObjectEncoder(CNN):
    def __init__(self):
        super(ObjectEncoder, self).__init__(channels=2,
                                            inputImageDimension=RESOLUTION)

    def forward(self, spec, obj):
        if isinstance(obj, list): # batched - expect a single spec and multiple objects
            # spec and object 
            spec = np.repeat(spec[np.newaxis,:,:],len(obj),axis=0)
            obj = np.stack(obj)
            return super(ObjectEncoder, self).forward(np.stack([spec, obj],1))
        else: # not batched
            return super(ObjectEncoder, self).forward(np.stack([spec, obj]))
        

class SpecEncoder(CNN):
    def __init__(self):
        super(SpecEncoder, self).__init__(channels=1,
                                          inputImageDimension=RESOLUTION)


"""Training"""
def randomScene(maxShapes=5, minShapes=1, verbose=False, export=None, no_repeat=True):
    # Choose number of shapes to include
    desiredShapes = np.random.randint(minShapes, maxShapes + 1)
    # Generate initial shape
    s=Primitive(np.random.randint(len(dsl.PRIMITIVES)))
    for _ in range(desiredShapes - 1):
        # Sample new primitive and add to list of arguments
        o = [s, Primitive(np.random.randint(len(dsl.PRIMITIVES)))]
        # Resample until new primitive until it's unique, if required
        if no_repeat:
            while o[1] in o[0]:
                o[1] = Primitive(np.random.randint(len(dsl.PRIMITIVES)))
        # Shuffle objects randomly, so which goes where is random
        np.random.shuffle(o)
        # Get shapes of rendered version of both, because that will constrain shift
        d = [i.render().shape for i in o]        
        # Randomly choosse to      
        if np.random.rand() > 0.5:
            # Horizontal concetanation, with random shift depending on heights
            s = Hor(o[0], o[1], shift=np.random.randint(-d[1][0] + 1, d[0][0]))
        else:
            # Vertical concetanation, with random shift depending on widths
            s = Vert(o[0], o[1], shift=np.random.randint(-d[1][1] + 1, d[0][1]))
    if verbose:
        print(s)
        print(ProgramGraph.fromRoot(s, oneParent=True).prettyPrint())
        import matplotlib.pyplot as plt
        plt.imshow(s.execute(), cmap='Greys', vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks([])
        plt.title(s)
        plt.show()
    if export:
        import matplotlib.pyplot as plt        
        plt.imshow(s.execute(), cmap='Greys', vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks([])        
        plt.title(s)        
        plt.savefig(export)
    return s

def trainCSG(m, getProgram, trainTime=None, checkpoint=None):
    print("cuda?",m.use_cuda)
    assert checkpoint is not None, "must provide a checkpoint path to export to"
    
    optimizer = torch.optim.Adam(m.parameters(), lr=0.001, eps=1e-3, amsgrad=True)
    
    startTime = time.time()
    reportingFrequency = 100
    totalLosses = []
    movedLosses = []
    iteration = 0

    while trainTime is None or time.time() - startTime < trainTime:
        s = getProgram()
        l = m.gradientStepTrace(optimizer, s.execute(), s.toTrace())
        totalLosses.append(sum(l))
        movedLosses.append(sum(l)/len(l))

        if iteration%reportingFrequency == 0:
            print(f"\n\nAfter {iteration} gradient steps...\n\tTrace loss {sum(totalLosses)/len(totalLosses)}\t\tMove loss {sum(movedLosses)/len(movedLosses)}\n{iteration/(time.time() - startTime)} grad steps/sec")
            totalLosses = []
            movedLosses = []
            with open(checkpoint,"wb") as handle:
                pickle.dump(m, handle)

        iteration += 1

def testCSG(m, getProgram, timeout, export):
    oneParent = m.oneParent
    solvers = [# RandomSolver(dsl),
               # MCTS(m, reward=lambda l: 1. - l),
               # SMC(m),
               ForwardSample(m, maximumLength=18)]
    loss = lambda spec, program: 1-max( o.IoU(spec) for o in program.objects() ) if len(program) > 0 else 1.

    testResults = [[] for _ in solvers]

    for _ in range(10):
        spec = getProgram()
        print("Trying to explain the program:")
        print(ProgramGraph.fromRoot(spec, oneParent=oneParent).prettyPrint())
        print()
        for n, solver in enumerate(solvers):
            testSequence = solver.infer(spec.execute(), loss, timeout)
            testResults[n].append(testSequence)
            for result in testSequence:
                print(f"After time {result.time}, achieved loss {result.loss} w/")
                print(result.program.prettyPrint())
                print()

    plotTestResults(testResults, timeout,
                    defaultLoss=1.,
                    names=[# "MCTS","SMC", 
                           "FS"],
                    export=export)

def plotTestResults(testResults, timeout, defaultLoss=None,
                    names=None, export=None):
    import matplotlib.pyplot as plot

    def averageLoss(n, T):
        results = testResults[n] # list of list of results, one for each test case
        # Filter out results that occurred after time T
        results = [ [r for r in rs if r.time <= T]
                    for rs in results ]
        losses = [ min([defaultLoss] + [r.loss for r in rs]) for rs in results ]
        return sum(losses)/len(losses)

    plot.figure()
    plot.xlabel('Time')
    plot.ylabel('Average Loss')

    for n in range(len(testResults)):
        xs = list(np.arange(0,timeout,0.1))
        plot.plot(xs, [averageLoss(n,x) for x in xs],
                  label=names[n])
    plot.legend()
    if export:
        plot.savefig(export)
    else:
        plot.show()
        
        
def debug(mode='demo'):
    import argparse
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("--checkpoint", default="checkpoints/tetris_test.pickle")
    parser.add_argument("--maxShapes", default=4,
                            type=int)
    parser.add_argument("--trainTime", default=None, type=float,
                        help="Time in hours to train the network")
    parser.add_argument("--attention", default=1, type=int,
                        help="Number of rounds of self attention to perform upon objects in scope")
    parser.add_argument("--heads", default=2, type=int,
                        help="Number of attention heads")
    parser.add_argument("--hidden", "-H", type=int, default=256,
                        help="Size of hidden layers")
    parser.add_argument("--timeout", default=5, type=float,
                        help="Test time maximum timeout")
    parser.add_argument("--oneParent", default=False, action='store_true')
    arguments = parser.parse_args()    
        
    if mode == "demo":
        for n in range(20):
            randomScene(export=f"tmp/tetris_{n}.png",maxShapes=arguments.maxShapes)
        import sys
        sys.exit(0)
    if mode == "train":
        m = ProgramPointerNetwork(ObjectEncoder(), SpecEncoder(), tDSL,
                                  oneParent=arguments.oneParent,
                                  attentionRounds=arguments.attention,
                                  heads=arguments.heads,
                                  H=arguments.hidden)
        trainCSG(m, lambda: randomScene(maxShapes=arguments.maxShapes),
                 trainTime=arguments.trainTime*60*60 if arguments.trainTime else None,
                 checkpoint=arguments.checkpoint)
    elif mode == "test":
        with open(arguments.checkpoint,"rb") as handle:
            m = pickle.load(handle)
        testCSG(m,
                lambda: randomScene(maxShapes=arguments.maxShapes, minShapes=arguments.maxShapes), arguments.timeout,
                export=f"figures/tetris_{arguments.maxShapes}_shapes.png")
  


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("mode", choices=["train","test","demo"])
    parser.add_argument("--checkpoint", default="checkpoints/tetris.pickle")
    parser.add_argument("--maxShapes", default=5,
                            type=int)
    parser.add_argument("--trainTime", default=None, type=float,
                        help="Time in hours to train the network")
    parser.add_argument("--attention", default=1, type=int,
                        help="Number of rounds of self attention to perform upon objects in scope")
    parser.add_argument("--heads", default=2, type=int,
                        help="Number of attention heads")
    parser.add_argument("--hidden", "-H", type=int, default=256,
                        help="Size of hidden layers")
    parser.add_argument("--timeout", default=5, type=float,
                        help="Test time maximum timeout")
    parser.add_argument("--oneParent", default=False, action='store_true')
    arguments = parser.parse_args()

    if arguments.mode == "demo":
        for n in range(20):
            randomScene(export=f"tmp/tetris_{n}.png",maxShapes=arguments.maxShapes)
        import sys
        sys.exit(0)
    if arguments.mode == "train":
        m = ProgramPointerNetwork(ObjectEncoder(), SpecEncoder(), tDSL,
                                  oneParent=arguments.oneParent,
                                  attentionRounds=arguments.attention,
                                  heads=arguments.heads,
                                  H=arguments.hidden)
        trainCSG(m, lambda: randomScene(maxShapes=arguments.maxShapes),
                 trainTime=arguments.trainTime*60*60 if arguments.trainTime else None,
                 checkpoint=arguments.checkpoint)
    elif arguments.mode == "test":
        with open(arguments.checkpoint,"rb") as handle:
            m = pickle.load(handle)
        testCSG(m,
                lambda: randomScene(maxShapes=arguments.maxShapes, minShapes=arguments.maxShapes), arguments.timeout,
                export=f"figures/tetris_{arguments.maxShapes}_shapes.png")
