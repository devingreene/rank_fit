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

''' From the appendix of [1] '''

ab_dict = [
[ 0,0.5 ],
[ 4,0.51 ],
[ 11,0.52 ],
[ 18,0.53 ],
[ 26,0.54 ],
[ 33,0.55 ],
[ 40,0.56 ],
[ 47,0.57 ],
[ 54,0.58 ],
[ 62,0.59 ],
[ 69,0.6 ],
[ 77,0.61 ],
[ 84,0.62 ],
[ 92,0.63 ],
[ 99,0.64 ],
[ 107,0.65 ],
[ 114,0.66 ],
[ 122,0.67 ],
[ 130,0.68 ],
[ 138,0.69 ],
[ 146,0.7 ],
[ 154,0.71 ],
[ 163,0.72 ],
[ 171,0.73 ],
[ 180,0.74 ],
[ 189,0.75 ],
[ 198,0.76 ],
[ 207,0.77 ],
[ 216,0.78 ],
[ 226,0.79 ],
[ 236,0.8 ],
[ 246,0.81 ],
[ 257,0.82 ],
[ 268,0.83 ],
[ 279,0.84 ],
[ 291,0.85 ],
[ 303,0.86 ],
[ 316,0.87 ],
[ 329,0.88 ],
[ 345,0.89 ],
[ 358,0.9 ],
[ 375,0.91 ],
[ 392,0.92 ],
[ 412,0.93 ],
[ 433,0.94 ],
[ 457,0.95 ],
[ 485,0.96 ],
[ 518,0.97 ],
[ 560,0.98 ],
[ 620,0.99 ],
[ 736,1 ]]

#
# [1] "Elo-rating as a tool in the sequential estimation of dominance rankings"
# by Paul C H Albers and Han de Vries Animal Behavior 2001, 61, 489-945
#
