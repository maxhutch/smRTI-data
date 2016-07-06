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

guess_thin = [1.0, 113.0, 1.0, 4.0, 1.0]
fix_thin = [1.0, 1./(2.*np.pi), 1.0, 1./(np.pi)]
bounds_thin_cma = [[0.05,   1.0, 0.05, 2.0, 0.1],
                   [2.00, 200.0, 10.0, 5.0, 3.0] ]

thin_scaling = [1.0, 100.0, 1.0, 0.0001, 1.0]

#mix_bounds_cma = [(0.0, 0.001, 0.9, 0.9), (20.0, 1000.0, 1.1, 1.1)]
#mix_bounds_cma = [(0.0,), (5.0,)]

def get_scaling(bounds):
    scaling_of_vars = np.ones(len(bounds[0]))
    for i in range(scaling_of_vars.shape[0]):
        scaling_of_vars[i] = (bounds[1][i] - bounds[0][i]) / 2.
    return scaling_of_vars

def merge_coef(var, fix):
    return [fix[0], var[0], var[1], var[2], fix[1], var[3], fix[2], var[4], fix[3]]

def f_dyn(t, y, C, Atwood, visc, L, mixed_fluid):
    M  = (C[4]*L + C[3] * abs(y[0])) * L * L
    V  = (C[6]*L + C[5] * abs(y[0])) * L * L
   
    if isinstance(mixed_fluid, float):
        mix = 1. - mixed_fluid / V
    else:
        mix = 1. - mixed_fluid(t) / V

    #mix = 1. - y[2] / (abs(y[0]) * L * L)
         
    F = C[0] * Atwood * L * L * y[0] * mix
    F += - C[1] * L * L * y[1] * abs(y[1])
    F += - C[2] * visc * abs(y[0]) * y[1] 

    return [y[1], F / M]

def full_f(t, y, C, Atwood, visc, L, diff, delta_i):
    mixed_fluid = mix_direct([C[5], C[6]], L, diff, delta_i, t, y[0])

    return f_dyn(t, y, [C[0], C[1], C[2], C[3], C[4], C[7], C[8]], Atwood, visc, L, mixed_fluid)

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

def dyn_model(C, Atwood, visc, L, mix, y0, times):
    r = ode(f_dyn).set_integrator('vode', method='bdf', atol=1.0e-6, with_jacobian=False)
    r.set_initial_value(y0, times[0]).set_f_params(C, Atwood, visc, L, mix)

    [T, H, v] = integrate(r, times)
    return T, H, v

from scipy.integrate import ode
def full_model(C, Atwood, visc, L, diff, delta_i, y0, times):

    r = ode(full_f).set_integrator('vode', method='bdf', atol=1.0e-6, with_jacobian=False)
    r.set_initial_value(y0, times[0]).set_f_params(C, Atwood, visc, L, diff, delta_i)
   
    [T, H, v] = integrate(r, times)
    MM = mix_direct([C[5], C[6]], L, diff, delta_i, T, H)

    return T, H, v, MM

def mix_direct(C, L, diff, delta_i, time, height):
    offset = delta_i**2. / (4. * diff)
    delta = 2. * np.sqrt((time+offset) * diff)
    diam = C[0] * L/(2.*4)
    #diam = L/(2.0)
    SA =  (C[1] * L + C[0]*height) * L
    integral = (
                 2*delta/np.sqrt(np.pi)*(1-np.exp(-(diam**2./np.square(delta)))) 
               + 2*diam*(1-erf(diam / delta))
               )
    DM = integral * SA
    return DM

def mix_model(C, L, diff, delta_i, times, h_func):
    return times, mix_direct(C, L, diff, delta_i, times, h_func(times))

from numpy.linalg import norm
def error(coeffs, Atwood, v, L, c, delta_i, y0, times, heights, mix):
    try:
        T, H, V, MM = full_model(coeffs, Atwood, v, L, c, delta_i, y0, times)
    except RuntimeError:
        return 1000.0
    se = 0.9 * np.sum(np.square(H - heights)) + 0.1 * np.sum(np.square(mix - MM)) / (L**4.)
    return np.sqrt(se / heights.size)

def both_error(coeffs, Atwood, v, L, c, delta_i, y0, times, heights, mix):
    try:
        T, H, V, MM = full_model(coeffs, Atwood, v, L, c, delta_i, y0, times)
    except RuntimeError:
        return 1000.0
    dyn_error = np.sqrt(np.sum(np.square(H - heights))/heights.size)
    mix_error = np.sqrt(np.sum(np.square(mix - MM)) / (L**4.) / heights.size)
    return dyn_error, mix_error

def dyn_error(coeffs, Atwood, v, L, times, height, mix):
    T, H, V = dyn_model(coeffs, Atwood, v, L, mix, times)

    se = np.sum(np.square(H - height))
    return np.sqrt(se / times.size)

def mix_error(coeffs, L, c, delta_i, times, h_func, mix):
    T, DM = mix_model(coeffs, L, c, delta_i, times, h_func)

    se = np.sum(np.square(mix - DM)) / (L**4.)
    return np.sqrt(se / times.size)

