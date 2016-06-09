import numpy as np

def f(t, y, C, Atwood, visc, L, diff):
    M = (C[4]*L + C[3] * abs(y[0])) * L * L
    SA = (C[6]*L*L + C[5] * L*abs(y[0]))
    
    delta = y[2] * np.sqrt(np.pi) / SA
    Ad = diff * SA  / delta
    
    mix = 1. - y[2] / M
            
    F = C[0] * Atwood * L * L * y[0] * mix
    F += - C[1] * visc * abs(y[0]) * y[1]
    F += - C[2] * L * L * y[1] * abs(y[1])
        

    return [y[1], F / M, Ad]

def integrate(r, t0, t1, dt):
    Nt = int((t1-t0) / dt) + 1
    Ny = len(r.y)
    soln = []
    for i in range(Ny):
        soln.append([r.y[i],])
    
    T = np.linspace(t0,t1,Nt)
    for t in T[1:]:
        r.integrate(t)
        for i in range(Ny):
            soln[i].append(r.y[i])
    soln_np = [T,] + [np.array(x) for x in soln]
    return soln_np

from scipy.integrate import ode
def model(C, Atwood, visc, L, diff, y0, t0, t1, dt):

    r = ode(f).set_integrator('vode', method='bdf', with_jacobian=False)
    r.set_initial_value(y0, t0).set_f_params(C, Atwood, visc, L, diff)
    
    [T, H, v, At] = integrate(r, t0, t1, dt)
    return T, H, v, At

from numpy.linalg import norm
def error(coeffs, Atwood, v, L, c, y0, times, heights):
    T, H, V, At = model(coeffs, Atwood, v, L, c, y0, times[0], times[-1], times[1] - times[0])
    return np.sqrt(np.sum(np.square(H - heights)) / heights.size)

