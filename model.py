import numpy as np
from scipy.special import erf

def filter_trajectory(times, heights, mass, L):
    stop = len(heights)
    for i in range(2,len(heights)):
        if heights[i] > 24.0:
            stop = i
            break
        if i + 1 < stop and all([heights[i] > heights[j] for j in range(i + 1, stop)]) :
            stop = i
            break

    return times[:stop], heights[:stop], mass[:stop]

guess = [1.0, 1.0, 113.0, 1.0, 1./(2*np.pi), 4.0, 1.0, 1.0, 1./(2*np.pi)]
bounds = ((1, 1), (0, 10), (0,400), (0.01, 100), (1./(2*np.pi), 1./(2*np.pi)), (0.01, 100), (1,1))

guess_thin = [1.0, 113.0, 1.0, 4.0, 1.0, 1.0]
bounds_thin = ((0, 10), (0,400), (0.01, 100), (0.01, 100), (0.01, 100), (0.01, 100))
fix_thin = [1.0, 1./(2.*np.pi), 1./(2.*np.pi)]
bounds_thin_cma = [(0.0,   0.0,  0.01,  0.0, 0.001,  0.001),
                   (4.0, 500.0, 100.0, 100.0, 1000.0, 1000.0) ]

#mix_bounds_cma = [(0.0, 0.001, 0.9, 0.9), (20.0, 1000.0, 1.1, 1.1)]
mix_bounds_cma = [(0.0, 0.001, ), (100.0, 1000.0)]

scaling_of_vars = np.ones(len(guess_thin))
for i in range(scaling_of_vars.shape[0]):
    scaling_of_vars[i] = (bounds_thin_cma[1][i] - bounds_thin_cma[0][i]) / 2.

scaling_of_mix = np.ones(len(mix_bounds_cma[0]))
for i in range(scaling_of_mix.shape[0]):
    scaling_of_mix[i] = (mix_bounds_cma[1][i] - mix_bounds_cma[0][i]) / 2.


def merge_coef(var, fix):
    return [fix[0], var[0], var[1], var[2], fix[1], var[3], var[4], var[5], fix[2]]

def mix_f(t, y, C, L, diff, h):
    SA = C[1]*(     L + C[0] * abs(h(t))) * L
    
    delta = y[0] * np.sqrt(np.pi) / SA
    Ad = diff * SA  /  (2.0 * np.sqrt(np.pi) * delta ) * (1.0 - np.exp(-(C[0] *L/(2.*np.pi))**2. / delta**2.))

    return [Ad,]

def full_f(t, y, C, Atwood, visc, L, diff):
    M  = (C[4]*L + C[3] * abs(y[0])) * L * L
    V  = C[6]*(C[8]*L + C[7] * abs(y[0])) * L * L
    SA = C[6]*(     L + C[5] * abs(y[0])) * L
    
    delta = y[2] * np.sqrt(np.pi) / SA
    Ad = diff * SA  /  (2.0 * np.sqrt(np.pi) * delta ) * (1.0 - np.exp(-(L/200.)**2. / delta**2.))
    
    mix = max(1. - y[2] / V, 0.0)
    #mix = 1. - y[2] / (abs(y[0]) * L * L)

            
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
def full_model(C, Atwood, visc, L, diff, y0, times):

    r = ode(full_f).set_integrator('vode', method='bdf', atol=1.0e-6, with_jacobian=False)
    r.set_initial_value(y0, times[0]).set_f_params(C, Atwood, visc, L, diff)
   
    [T, H, v, At] = integrate(r, times)
    return T, H, v, At

def mix_model(C, L, diff, y0, times, h_func):

    r = ode(mix_f).set_integrator('vode', method='bdf', atol=1.0e-6, with_jacobian=False)
    r.set_initial_value(y0[2:], times[0]).set_f_params(C, L, diff, h_func)
   
    [T, At] = integrate(r, times)
    return T, At

def mix_direct(C, L, diff, delta_i, time, height):
    offset = delta_i**2. / (4. * diff)
    delta = 2. * np.sqrt((time+offset) * diff)
    diam = C[0] * L/(2.*np.pi)
    SA = C[1] * (L + C[0]*height) * L
    integral = (
                 2*delta/np.sqrt(np.pi)*(1-np.exp(-(diam**2./np.square(delta)))) 
               + 2*diam*(1-erf(diam / delta))
               )
    DM = integral * SA
    return DM

def mix_model_direct(C, L, diff, y0, times, h_func):
    delta_i = y0[0]* np.sqrt(np.pi) / (2 * L**2.) 
    return times, mix_direct(C, L, diff, delta_i, times, h_func(times))

from numpy.linalg import norm
def error(coeffs, Atwood, v, L, c, y0, times, heights, mix):
    try:
        T, H, V, MM = full_model(coeffs, Atwood, v, L, c, y0, times)
    except RuntimeError:
        return 1000.0
    se = 0.9 * np.sum(np.square(H - heights)) + 0.1 * np.sum(np.square(mix - MM)) / (L**4.)
    return np.sqrt(se / heights.size)

def mix_error(coeffs, L, c, y0, times, h_func, mix):
    try:
        T, MM = mix_model(coeffs, L, c, y0, times, h_func)
    except RuntimeError:
        return 1000.0
    se = np.sum(np.square(mix - MM)) / (L**4.)
    return np.sqrt(se / times.size)

def mix_error_direct(coeffs, L, c, y0, times, h_func, mix):
    T, DM = mix_model_direct(coeffs, L, c, y0, times, h_func)

    se = np.sum(np.square(mix - DM)) / (L**4.)
    return np.sqrt(se / times.size)

