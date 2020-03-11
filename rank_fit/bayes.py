# An algorithm for assigning numerical ranks from a set of pairwise contests
# Copyright (C) 2018 Devin Greene
# email: devin@greene.cz

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from scipy.stats import norm
from functools import lru_cache
sq2 = np.sqrt(2)
invpi = 1/np.pi

# TODO The interface between the structure here and those of our
# methods is a mess.  We need to clean this up!

__all__ = ['obj_func', 'jac', 'hessian', 'recipsigmoid', 'drecipsigmoid',
        'ddrecipsigmoid', 'arctan', 'darctan', 'ddarctan']

# Kernel functions

def recipsigmoid(x):
    
    return 1+np.exp(-x)

def drecipsigmoid(x):

    return -np.exp(-x)

def ddrecipsigmoid(x):

    return np.exp(-x)

def arctan(x):

    return 0.5 + invpi*np.arctan(x)

def darctan(x):

    return invpi*1/(1+x**2)
    
def ddarctan(x):

    return -2*x*invpi/(1+x**2)**2

def obj_func(x,outcomes,alpha,func=recipsigmoid):
    ''' Basic function to minimize.  Uses reciprical of sigmoid as kernel.'''

    assert outcomes.shape == (len(x),)*2
    x = np.array(x)
    return (np.log(func(np.subtract.outer(x,x)))*outcomes).sum() + \
            alpha*(x**2).sum()
            
@lru_cache()
def _make_frame(n):
    ''' Helper function for computing Jacobian and Hessian '''
    # Broadcast magic
    frame1 = np.broadcast_to(np.eye(n)[...,None],(n,)*3)
    frame2 = -frame1.transpose(0,2,1)
    frame = frame1 + frame2
    return frame

def jac(x,outcomes,alpha,func=recipsigmoid,dfunc=drecipsigmoid):
    ''' The Jacobian of obj_func '''

    lenx = len(x)
    assert outcomes.shape == (lenx,)*2
    x = np.array(x)
    diffs = np.subtract.outer(x,x)
    ent = dfunc(diffs)/func(diffs)
    frame = _make_frame(lenx)

    return (outcomes*ent*frame).sum((1,2)) + 2*alpha*x

def hessian(x,outcomes,alpha,func=recipsigmoid, dfunc=drecipsigmoid, 
        ddfunc= ddrecipsigmoid):

    lenx = len(x)
    assert outcomes.shape == (lenx,)*2
    x = np.array(x)
    n = lenx
    diffs = np.subtract.outer(x,x)
    ent = ddfunc(diffs)/func(diffs) -\
            (dfunc(diffs)/func(diffs))**2

    frame = _make_frame(n)

    return np.tensordot(outcomes*ent*frame,frame,((1,2),)*2) - \
            2*alpha*np.eye(n)
