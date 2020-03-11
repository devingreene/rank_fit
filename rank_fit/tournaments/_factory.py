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

''' Used to simulate a tournament upon which our methods can be tested '''

__all__ = [ 'Tournament', 'RandomTournament',
        'EvenlyDistributedTournament', 
        'UserDefinedTournament',
        'TournamentFromHistory']

import numpy as np
# TODO Access this through package instead?
from rank_fit.abilities._factory import validate_ptable

class Tournament:
    ''' Base class for tournaments '''

    def __init__(self,ptable,ngames,keep_history = False):
        validate_ptable(ptable)
        self.ptable = ptable
        self.nplayers = ptable.shape[0]
        self.ngames = ngames
        # We don't make an outcomes attribute here

        # TODO Cleaner way of reporting when user tries to use this
        # base class

        # Elo_ranking needs entire history of tournament
        if keep_history:
            self.history = []

    def __repr__(self):
        return 'Tournament Outcomes\n'+self.outcomes.__repr__()

class RandomTournament(Tournament):
    ''' Select random pairs for contests and record results in an
    outcome matrix '''

    def __init__(self,*args,**kwargs):

        super().__init__(*args,**kwargs)

        # Checked that it's a square in base class ...
        dim = self.ptable.shape[0]

        # XXX Given the way np.mulitnomial works, it is _vital_ that
        # these probabilities add up to one!  Since we are zero-ing
        # the diagonal, we use the (dim-1)'th triangular number instead of
        # the dim'th.
        probs = 2*self.ptable/(dim*(dim-1))

        # Zeroize the diagonal
        probs[np.diag_indices(dim)] = 0

        self.outcomes = \
                np.random.multinomial(self.ngames,probs.flat).reshape(dim,dim)

        # Now create history if needed

        if hasattr(self,'history'):
            assert self.history == []

            for ind1,ind2 in np.ndindex(*(dim,)*2):
                self.history.extend(self.outcomes[ind1,ind2]*[[ind1,ind2]])

            np.random.shuffle(self.history)

class EvenlyDistributedTournament(Tournament):
    ''' Number of games is as much as possible evenly distributed
    between pairs of players.  Each pair plays at least 

    ngames // #pairs of players 

    games.  Any remainder is distributed randomly at above '''

    def __init__(self,*args,**kwargs):

        super().__init__(*args,**kwargs)

        n = self.nplayers
        npairs = n*(n-1)//2
        per_pair,rest = divmod(self.ngames,npairs)

        per_pair = np.full(npairs,per_pair) + \
                np.random.multinomial(rest,
                        np.full(npairs,1/npairs))

        if hasattr(self,'history'):
            assert not self.history

        self.outcomes = np.empty((n,n),int)

        I = np.indices(self.outcomes.shape)
        ltri = I[0] > I[1]
        utri = I[0] < I[1]

        # TODO There must be slicker way of doing this...
        self.outcomes[ltri] = np.random.binomial(per_pair,
                self.ptable[ltri])
        tr = self.outcomes.T
        tr[ltri] = per_pair - self.outcomes[ltri]
        self.outcomes[utri] = tr.T[utri]
        self.outcomes[np.diag_indices(n)] = 0

        # Create history if needed

        if hasattr(self,'history'):
            assert self.history == []

            for ind1,ind2 in np.ndindex(self.outcomes.shape):
                self.history.extend(
                        self.outcomes[ind1,ind2]*[[ind1,ind2]])

            np.random.shuffle(self.history)

class UserDefinedTournament(Tournament):
    ''' The user provides her own outcomes matrix '''

    def __init__(self,outcomes,**kwargs):

        # TODO Fix this nonsense later
        
        # Dummy arguments
        args = [ np.full(outcomes.shape,np.nan),
                #ngames
                outcomes.sum() ]

        super().__init__(*args,**kwargs)

        self.outcomes = outcomes

        #Let's allow a history so it works with Elo

        if hasattr(self,'history'):
            #Make a fake history
            assert self.history == []

            for ind1,ind2 in np.ndindex(self.outcomes.shape):
                self.history.extend(
                        self.outcomes[ind1,ind2]*[[ind1,ind2]])

            np.random.shuffle(self.history)

class TournamentFromHistory(Tournament):
    ''' Make tournament from user provided history '''

    def __init__(self,history,**kwargs):
        assert isinstance(history,(list,tuple))
        dim = 0
        for p in history:
            assert len(p) == 2
            dim = max(dim,*p)
        dim += 1

        outcomes = np.empty((dim,dim))

        history_t = [ tuple(p) for p in history ]
        for index in np.ndindex(outcomes.shape):
            outcomes[index] = history_t.count(tuple(index))

        # TODO Fix this nonsense later
        
        # Dummy arguments
        args = [ np.full(outcomes.shape,np.nan),
                #ngames
                outcomes.sum() ]

        super().__init__(*args,**kwargs)

        self.outcomes = outcomes

        if hasattr(self,'history'):
            assert self.history == []

            self.history = history
