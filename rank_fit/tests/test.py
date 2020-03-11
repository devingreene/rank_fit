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

''' Test suite '''

import numpy as np
from rank_fit.abilities import *
from rank_fit.tournaments import *
from rank_fit.rankers import ( PBS_ranker,glicko_ranker,elo_ranker,
        ConvergenceFailure )
from rank_fit.bayes import *
from scipy.optimize import minimize

__all__ = [ 'doit' ]

def doit():
    # Run the gamut

    skills = [ [1,2,3],
            [ 1,2,5,20],
            [-10,-9,-8,1,2 ],
            [ 0.1,0.2,0.3,0.4],
            [5,1,4,2,3,3,0],
            [1,1,1,1] ]

    for s in skills:
        for ptable in [standard(s),interactions(s,'ant'),
                interactions(s,'syn')]:
            for alpha in [1e-4,1e-2,1,10]:
                for tourtype in [ RandomTournament,
                        EvenlyDistributedTournament ]:
                    for ranker in [ PBS_ranker,elo_ranker,
                            glicko_ranker ]:
                        print('With\n'+
                                ('{}\n'*4).format(ptable,alpha,tourtype,ranker))
                        kh = ranker is elo_ranker
                        alpha_kw = {'alpha':alpha} \
                                if ranker is PBS_ranker else {}
                        k_kw = {'k':32} \
                                if ranker is elo_ranker else {}
                        tour = tourtype(ptable,ngames=5,keep_history = kh)
                        tour = tourtype(ptable,ngames=200,keep_history=kh)
                        try:
                            ranker(tour,**alpha_kw,**k_kw)
                        except ConvergenceFailure as e:
                            print('ConvergeFailure with {}\n'\
                                    'message: {}'.\
                                    format((s,ptable,alpha,tourtype,tour),e))

    # Test TournamentFromHistory

    tour = TournamentFromHistory(\
            [[i,i] for i in range(0,20)],\
            keep_history=True)
    if not ( hasattr(tour,'history') or \
            tour.history !=  [[i,i] for i in range(0,20)] or \
            outcomes.shape[0] != outcomes.shape[1] or \
            tour.outcomes != np.eye(outcomes.shape[0]) ):
        raise Exception('Problem with `TournamentFromHistory\'')

    xs = [[1,2,3],[2,1,-5],[-1e-6,0.1,0.2],[0,0,0]]
    oc = np.array([[1,2,3],[4,5,6],[-2,0,-.001]])

    tour = UserDefinedTournament(oc)
    oc = tour.outcomes

    from functools import partial
    def defaults(f):
        return partial(f,outcomes=oc,alpha=0.01)

    for x in xs:
        defaults(obj_func)(x,func=recipsigmoid)
        defaults(obj_func)(x,func=arctan)
        defaults(jac)(x,func=recipsigmoid,dfunc=drecipsigmoid)
        defaults(jac)(x,func=arctan,dfunc=darctan)
        defaults(hessian)(x,func=recipsigmoid,dfunc=drecipsigmoid,ddfunc=ddrecipsigmoid)
        defaults(hessian)(x,func=arctan,dfunc=darctan,ddfunc=ddarctan)

        # Jacobian coherency
        minimize(obj_func,[0]*len(x),(oc,0.01),jac=jac)
        minimize( partial(obj_func,func=arctan),
            [0]*len(x),(oc,0.01), 
            jac = partial(jac,func=arctan,dfunc=darctan))

doit()
