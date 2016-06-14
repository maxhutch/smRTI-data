import numpy as np

def f(t, y, C, Atwood, visc, L, diff):
    M = (C[4]*L + C[3] * abs(y[0])) * L * L
    SA = (C[6]*L*L + C[5] * L*abs(y[0]))
    
    delta = y[2] * np.sqrt(np.pi) / SA
    Ad = diff * SA  / delta
    
    mix = 1. - y[2] / M
            
    F = C[0] * Atwood * L * L * y[0] * mix
    F += - C[1] * L * L * y[1] * abs(y[1])
    F += - C[2] * visc * abs(y[0]) * y[1]
        

    return [y[1], F / M, Ad]

def integrate(r, times):
    Ny = len(r.y)
    soln = []
    for i in range(Ny):
        soln.append([r.y[i],])
    
    for t in times[1:]:
        r.integrate(t)
        if not r.successful():
            raise RuntimeError("integration failed")
        for i in range(Ny):
            soln[i].append(r.y[i])
    soln_np = [times,] + [np.array(x) for x in soln]
    return soln_np

from scipy.integrate import ode
def model(C, Atwood, visc, L, diff, y0, times):

    r = ode(f).set_integrator('vode', method='bdf', atol=1.0e-6, with_jacobian=False)
    r.set_initial_value(y0, times[0]).set_f_params(C, Atwood, visc, L, diff)
   
    [T, H, v, At] = integrate(r, times)
    return T, H, v, At

from numpy.linalg import norm
def error(coeffs, Atwood, v, L, c, y0, times, heights):
    try:
        T, H, V, At = model(coeffs, Atwood, v, L, c, y0, times)
    except RuntimeError:
        return 1000.0
    return np.sqrt(np.sum(np.square(H - heights)) / heights.size)

