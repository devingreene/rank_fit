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
from functools import lru_cache
from scipy.stats import norm,uniform
sq2 = np.sqrt(2)

__all__ = ['standard','interactions']

marr = np.MachAr()

class InvalidPtable(Exception):
    pass

def _pbalance(A,B):
    '''Copy the lower triangular part of B to that of A, 0.5 on the
    diagonal, and A_ij = 1 - A_ji for i < j'''

    assert A.shape == B.shape
    I = np.indices(A.shape)
    ltri = I[0] > I[1]
    utri = I[1] > I[0]

    A[ltri] = B[ltri]
    np.fill_diagonal(A,0.5)
    A[utri] = 1 - A.T[utri]

def validate_ptable(ptable):

    if not isinstance(ptable,np.ndarray):
        raise InvalidPtable

    if not ( ptable.shape[-2] == ptable.shape[-1]):
        raise InvalidPtable

    # Okay if dummy ptable via UserDefinedTournament
    if np.isnan(ptable).all():
        return

    if ( abs( ptable + ptable.T - 1) >= 
            1.667*marr.eps).all():
        raise InvalidPtable

def standard(skills,cdf=None):
    ''' Produce of matrix of win probabilities based on the skill levels
    of the players.  Basically p = cdf(playerA_skill - playerB_skill) '''

    if cdf is None:
        cdf = norm(scale=sq2).cdf

    ptable = np.empty((len(skills),)*2)
    _pbalance(ptable,cdf(np.subtract.outer(skills,skills)))

    validate_ptable(ptable)

    return ptable

def interactions(skills,style,a=0.5,cdf=None):
    ''' Produce "exotic" win probabilities.  This is mostly for testing
    robustness of ranking methods '''

    @lru_cache()
    def assign(i,j):
        if i <= j: return 0
        if i == j + 1: return ptable[i,j]
        return wedge(i,j)

    if a < 0:
        raise ValueError("`a' must be a positive number")

    if style == 'ant':
        def wedge(x,y):
            A,B = sorted([assign(x-1,y),assign(x,y+1)])
            return uniform.rvs(B,a*abs(A)+B)

    elif style == 'syn':
        def wedge(x,y):
            return assign(x-1,y) + assign(x,y+1) + uniform.rvs(0,a)

    else:
        raise Exception('We have not yet defined styles other than '\
                '\'antagonistic\'(\'ant\') and \'synergistic\'(\'syn\')')

    nplayers = len(skills)
    ptable = np.zeros((nplayers,)*2)

    # TODO Use numpy's index tricks for this?
    I = np.indices(ptable.shape)
    ptable[I[0] == I[1] + 1] = np.diff(skills)
    
    for ind in np.ndindex(ptable.shape):
        if ind[0] > ind[1] + 1:
            ptable[ind] = assign(*ind)

    if cdf is None:
        cdf = norm(scale=sq2).cdf

    ptable = cdf(ptable)

    _pbalance(ptable,ptable)

    validate_ptable(ptable)

    return ptable

# vim: tw=70
