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
from warnings import warn,filterwarnings

__all__ = ['glicko_update']

filterwarnings('error',category=RuntimeWarning)

def glicko_update(r,rd,outcomes):
    ''' See [1] and [2]. Players are assumed to start at 
    r=1500 and rd=350 if previously unrated '''

    # Following comments refer to corresponding notation used in [2]

    # \mu
    r = np.array(r)

    # \sigma
    rd = np.array(rd)

    nplayers = len(r)

    assert nplayers == len(rd) == outcomes.shape[0] == outcomes.shape[1]

    # q = ln(10)/400
    q = 0.0057564627324851146

    # n_j
    ngames = outcomes + outcomes.T

    # g(\sigma^2)^2
    grd2 = 1/(1+3*q**2*rd**2/np.pi**2)

    # g(\sigma^2)
    grd = np.sqrt(grd2)

    # outcomes = sum_k s_ijk

    # E(s|\mu,\mu_j,\sigma_j)
    E = 1/(1+10**(-grd*np.subtract.outer(r,r)/400))

    # \delta^2
    with np.errstate(divide='raise'):
        try: 
            d2 = 1/(q**2*(ngames*grd2*E*(1-E)).sum(1))
        # In case of zero divide, non-participants' skill ratings stay the same,
        # and ranking procedure carried out with others.
        except FloatingPointError:
            part= []
            for i,row in enumerate(ngames):
                if (row != 0).any():
                    part.append(i)
            
            ix = np.ix_(part,part)
            tmp = glicko_update(r[part],
                    rd[part],
                    outcomes[ix])
            r[part],rd[part] = tmp
            return r,rd

    # \sigma'^2
    new_rd = np.sqrt(1/(1/rd**2 + 1/d2))

    # \mu'
    new_r = r + q/(1/rd**2 + 1/d2)*(grd*(outcomes - ngames*E)).sum(1)

    return new_r,new_rd

# [1] http://www.glicko.net/glicko/glicko.pdf
# 
# [2] "Parameter estimation in large dynamic paired comparison experiments"
#     James Glickman, Applied Statistics (1999) 48, Part 3, pp. 377-394
