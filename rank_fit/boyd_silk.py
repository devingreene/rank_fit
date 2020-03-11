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

def boyd_silk_update(p,outcomes):
    ''' Iteration step in [1] p49 '''

    return outcomes.sum(1)/\
            ((outcomes + outcomes.T)/\
            np.add.outer(p,p)).sum(1)

# [1] "A Method for Assigning Cardinal Dominance Ranks" 
#     Robert Boyd and Joan B. Silk, Anim. Behav., 1983, 31, 45-58
# 
# vim: tw=70
