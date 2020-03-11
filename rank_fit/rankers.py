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

''' The ranking algorithms considered in our comparisons '''

import numpy as np
from .bayes import obj_func,jac
from scipy.optimize import minimize
from warnings import warn,filterwarnings
from functools import partial
from .elo.elo import ab_fun
from .glicko import glicko_update
from .boyd_silk import boyd_silk_update

__all__ = [ 'PBS_ranker', 'glicko_ranker', 'elo_ranker',
        'boyd_silk_ranker' ]

class ConvergenceFailure(Exception):
    pass

def PBS_ranker(tournament,alpha,start = None,obj_func_args={},*args,**kwargs):
    '''Generate the rankings based on a series of games and a starting value
    '''
    
    # Start optimization at origin by default
    # ( got any better ideas? )
    if start is None: 
        start = np.zeros(tournament.nplayers)

    outcomes = tournament.outcomes

    obj_f = partial(obj_func,func=obj_func_args['func']) \
            if obj_func_args else obj_func

    if 'hess' in kwargs:
        method_kw = {'method':'newton-cg'}
    else:
        method_kw = {}

    try:
        with np.errstate(all='raise'):
            res = minimize(obj_f,start,(outcomes,alpha),jac=jac, 
                            **method_kw)

    except FloatingPointError as e:
        raise ConvergenceFailure(e)

    if not res.success:
        raise ConvergenceFailure('Convergence failure: '+res.message)

    return res.x

def glicko_ranker(tournament,*args,**kwargs):
    outcomes = tournament.outcomes
    nplayers = tournament.nplayers
    r,_ = glicko_update([1500]*nplayers,[350]*nplayers,outcomes)

    return r

def elo_ranker(tournament,k,use_pretourn_scores=False):

    nplayers = tournament.nplayers
    rank = np.zeros(tournament.nplayers)

    if use_pretourn_scores:
        outcomes = tournament.outcomes
        assert (nplayers,)*2 == outcomes.shape
        # We are assuming that pre-tournaments scores are all equal.
        # So the weight for _any_ individual contest will be 0.5
        return 0.5*k*(outcomes.sum(1) - outcomes.sum(0))

    if not hasattr(tournament,'history'):
        raise Exception('Elo ranking requires tournament history')

    history = tournament.history

    for a,b in history:
        # Recall that *a* beat *b*

        d = rank[a] - rank[b]

        # Update ranking
        rank[a] += k*ab_fun(d)
        rank[b] -= k*ab_fun(d)

    return rank

def boyd_silk_ranker(tournament,tol = 0.5e-3,
        iterlimit=10000,*args,**kwargs):
    ''' We use the algorithm defined in [1] to rank the players '''

    # [1] has a weird recommended start.  To quote:
    #
    # "We have found that setting P^{\hat}_i equal to individual i's
    # ordinal rank divided by t  works well."  
    #
    # (Here t = tourament.nplayers) 
    #
    # But isn't the rank what we are trying to find?  Do they have a
    # prior in mind?  Befuddled, I set them all equal to
    # 1/tournament.nplayers.

    nplayers = tournament.nplayers
    p = np.array([1/tournament.nplayers]*nplayers)

    # p49 in [1]
    try:
        for _ in range(iterlimit):
            nextp = boyd_silk_update(p,tournament.outcomes)
            if abs(nextp - p).sum() <= tol:
                p = nextp
                p = p/p.sum()
                break
            p = nextp

            # This isn't explicitly written out in [1]
            p = p/p.sum()
        else:
            raise ConvergenceFailure('Iteration limit exceeded')
    except RuntimeWarning as e:
        raise ConvergenceFailure(e)


    # Find Ds as per formula (2) on p48 in [1], but with our
    # modification that the mean is zero and higher => better
    res = np.log(p) 
    return res - res.mean()

# [1] "A Method for Assigning Cardinal Dominance Ranks" 
#     Robert Boyd and Joan B. Silk, Anim. Behav., 1983, 31, 45-58
# 
# vim: tw=70
