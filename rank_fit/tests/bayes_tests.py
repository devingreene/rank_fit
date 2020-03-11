''' Tests for functions in bayes.py '''

import numpy as np
from rank_fit.bayes import *
import warnings

def warning(func,_x,_dx):
    warnings.warn("High term with {} at \n_x={} and \n_dx={}".
            format(func.__name__,_x,_dx))

# Derivatives

x = np.array([[0.1, 0.5, 1.0, 1.2],
    [0.2,1.0,2.0,2.4],
    [0.3,0.1,0.5,-0.2],
    [-0.2,-0.3,-1.2,-2.4]])
#Try different orders of magnitude
dx = np.dot(np.diag([ 1e-6, 1e-7, 1e-8, 1e-9 ]),x)

def testfun(fun,dfun,ddfun):
    for _x in x:
        for _dx in dx:
            o2 = fun(_x+_dx) - fun(_x) - dfun(_x)*_dx
            do2 = dfun(_x+_dx) - dfun(_x) - ddfun(_x)*_dx
            if sum(abs(o2)**2) >= 0.1667*sum(_dx**2)**0.5:
                warning(fun,_x,_dx)
            if sum(abs(do2)**2) >= 0.1667*sum(_dx**2)**0.5:
                warning(dfun,_x,_dx)

outcomes = np.array([1,2,3]*5+[1]).reshape(4,4)

def testjachess(fun,dfun,ddfun):
    alpha= 0.01
    obj_f = lambda x: obj_func(x,outcomes,alpha,func=fun)
    jac_ = lambda x: jac(x,outcomes,alpha,func=fun,dfunc=dfun)
    hess_ = lambda x: hessian(x,outcomes,alpha,func=fun,dfunc=dfun,ddfunc=ddfun)

    for _x in x:
        for _dx in dx:
            o2 = obj_f(_x+_dx) - obj_f(_x) - np.dot(jac_(_x),_dx)
            do2 = jac_(_x+_dx) - jac_(_x) - np.dot(hess_(_x),_dx)
            if abs(o2) >= 0.1667*sum(_dx**2)**0.5:
                warning(fun,_x,_dx)
            if sum(do2**2)**0.5 >= 0.1667*sum(_dx**2)**0.5:
                warning(dfun,_x,_dx)

testfun(recipsigmoid,drecipsigmoid,ddrecipsigmoid)
testfun(arctan,darctan,ddarctan)
testjachess(recipsigmoid,drecipsigmoid,ddrecipsigmoid)
testjachess(arctan,darctan,ddarctan)
