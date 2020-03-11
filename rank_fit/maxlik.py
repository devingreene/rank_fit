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

''' Consider N items each with a latents 'quality' score.  We have a set of 
independent comparisons, and fit a maximum likelihood model.  '''

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from functools import partial

__all__ = [ 'default_kernel', 'default_dkernel', 'default_ddkernel',
        'objfunc', 'jac', 'hess', 'optimize' ]

cdf = norm.cdf
pdf = norm.pdf
fac = np.sqrt(2)

def default_kernel(X):
    return cdf(X/fac)

def default_dkernel(X):
    return pdf(X/fac)/fac

def default_ddkernel(X):
    return -X*pdf(X/fac)/fac**3

def objfunc(x,data,kernel=default_kernel):
    ''' Objective function.  `x0' will be fixed to zero '''
    x = np.r_[0,x]
    return (-data*np.log(kernel(np.subtract.outer(x,x)))).sum()

def jac(x,data,kernel=default_kernel,dkernel = default_dkernel):
    ''' Jacobian for the kernel '''
    x = np.r_[0,x]
    arg = np.subtract.outer(x,x)
    ker,dker = kernel(arg),dkernel(arg)
    e = dker/ker
    return np.array( [ -(data[i,:]*e[i,:]).sum()
        + (data[:,i]*e[:,i]).sum() 
        for i in range(1,len(x)) ] )

def hess(x,data,kernel=default_kernel,dkernel=default_dkernel,
        ddkernel=default_ddkernel):
    ''' Hessian for the kernel '''
    x = np.r_[0,x]
    arg = np.subtract.outer(x,x)
    ker,dker,ddker = kernel(arg),dkernel(arg),ddkernel(arg)
    e = ( dker**2/ker - ddker )/ker
    return np.array( [ [ -data[i,j]*e[i,j] - data[j,i]*e[j,i] \
            if i != j else \
            (data[i,:]*e[i,:]).sum() + (data[:,i]*e[:,i]).sum()
            for j in range(1,len(x)) ] for i in range(1,len(x))] )
        
def optimize(data,objfunc=partial(objfunc,kernel=default_kernel),
        *args,**kwargs):
    '''  Use scipy's `minimize' in `optimization' '''

    x0 = np.zeros(len(data)-1)
    res = minimize(objfunc,x0,(data,),*args,**kwargs)
    print(res)
    return np.concatenate(([0],res.x))

